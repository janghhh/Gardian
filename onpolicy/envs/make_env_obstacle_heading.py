import gymnasium as gym
import numpy as np
import airsim
import math
import time
import random



class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",
        follower_names=("Follower0", "Follower1", "Follower2"),
        port=41451,
        step_length=0.01,
        leader_velocity=1.0,
        optimal_distance=10.0,
        far_cutoff=60.0,
        too_close=0.5,
        dt=0.1,
        do_visualize=True
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]

        # 충돌 관련 설정
        self.COLLISION_THRESHOLD = 1.5
        self.STOP_DISTANCE_LEADER_OBSTACLE = 1.0
        
        # 속도/액션 버퍼
        self.vmax_self = 2.0
        self._timestep = float(dt)   # dt와 맞춰주거나, 쓰지 않으면 제거해도 OK

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}
        self._last_time = {}

        # 액션 버퍼
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = -10.0
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)
        self.max_cmd_speed = self.step_length / self.dt
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)

        # States
        self.step_count = 0
        self.episode_count = 0

        # 동적 장애물 관련
        self.isIdle = None
        self.D_O_STATE = {0: "idle", 1: "attack"}
 
        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1  # 나를 제외한 아군 수
        self.K_enemy = 1                       # 동적 장애물 1대
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy
        
        # ---- 관측 공간 범위 정의 ----
        low_bearing = -1.0
        high_bearing = 1.0
        low_dist = 0.0
        high_dist = 200.0
        
        # ★ 실제 넣는 값은 [-1,1] 이므로 여기도 맞춰줌
        low_vel = -1.0      # 정규화된 closing_speed_norm
        high_vel = 1.0
        low_rate = -1.0     # 정규화된 los_rate_norm
        high_rate = 1.0

        # ★ 역할 플래그: 0(defender) ~ 1(interceptor)
        low_role = 0.0
        high_role = 1.0

        # [리더(2)] + [아군(2)*K] + [적(4)*K] + [role_flag(1)]
        per_agent_low = (
            [low_bearing, low_dist] +
            [low_bearing, low_dist] * self.num_ally +
            [low_bearing, low_dist, low_vel, low_rate] * self.num_enemy +
            [low_role]
        )
        per_agent_high = (
            [high_bearing, high_dist] +
            [high_bearing, high_dist] * self.num_ally +
            [high_bearing, high_dist, high_vel, high_rate] * self.num_enemy +
            [high_role]
        )

        obs_dim = len(per_agent_low)
        share_obs_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low, dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.MAX_YAW = math.radians(90)
        self.MAX_SPEED = 10

        self.action_spaces = {
            agent: gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float64,
            )
            for agent in self.possible_agents
        }

        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,),
            dtype=np.float32,
        )

        self.dynamic_name = "DynamicObstacle"
        # PN 보상용 버퍼들
        self._prev_d_leader_enemy = None
        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._prev_los_angle = {a: None for a in self.possible_agents}

        # PN-style reward parameters
        self.REWARD_HIT_ENEMY = 120.0
        self.REWARD_LEADER_HIT = -300.0
        self.REWARD_AGENT_CRASH = -80.0

        self.W_CLOSE = 1.0
        self.W_LOS = 1.0
        self.W_DIST = 1.0

        self.MAX_DIST_DELTA = 5.0
        self.MAX_ANGLE_DELTA = math.radians(20.0)    
        self._interceptor = None             
        self.W_DEF_LEADER   = 0.2   # 수비 보상 최대값 (너무 크지 않게)
        self.DEF_LEADER_BAND = 5.0  # [optimal_distance ± 5m] 안이면 보상
        self.W_ALIGN = 1.0

        # 위치 캐시
        self.start_location = {}
        self.current_location = {}

        # 클라이언트 셋업
        self.client = airsim.MultirotorClient(ip=ip_address, port=port)
        self.client.confirmConnection()
        
        self._last_visualize_t = time.time()


    # ======================================================================
    # 헬퍼 메서드: 포즈/속도/관측 관련
    # ======================================================================
    def _angle_and_distance(self, src_drone, target_drone):

        # 두 지점의 X축과 Y축의 변화량
        dx = float(
            self.current_location[target_drone].position.x_val - 
            self.current_location[src_drone].position.x_val
        )
        dy = float(
            self.current_location[target_drone].position.y_val - 
            self.current_location[src_drone].position.y_val        
        )
        
        # 내가 쓰는 heading_state 참조
        src_yaw = self.heading_state[src_drone]

        # 상대 거리 구하기 (피타고라스, World Frame 기준)
        distance_diff = math.sqrt(dx**2 + dy**2)

        # 상대 방위 구하기
        ## - World Frame 기준 두 좌표의 상대 방위 구하기 (arctan 활용)
        _angle = math.atan2(dy, dx) # 축이 90° 틀어져 있음 dx, dy -> dy, dx로 수정

        ## - 두 방위를 빼면 드론 기준 상대 방위를 할 수 있다. (절대 방위 차 - 현재 드론의 방위)
        angle_diff = ((_angle - src_yaw) + math.pi) % (2 * math.pi) - math.pi   # 각도 정규화 (-180, +180)으로 정규화
    
        return angle_diff, distance_diff

    def _get_current_location(self):
        self.current_location = {}  # Init
        self.current_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.current_location[agent] = self.client.simGetObjectPose(agent)
        self.current_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

    def _update_roles(self):
        """
        현재 step에서 '적과 가장 가까운 에이전트'를 interceptor로 지정해서
        self._interceptor 에 저장.
        """
        enemy_pose = self.current_location[self.dynamic_name].position
        enemy_xy = np.array([enemy_pose.x_val, enemy_pose.y_val], dtype=np.float32)

        min_agent = None
        min_dist = float("inf")
        for ag in self.possible_agents:
            pose = self.current_location.get(ag, None)
            if pose is None:
                continue
            ag_xy = np.array([pose.position.x_val, pose.position.y_val], dtype=np.float32)
            d = float(np.linalg.norm(ag_xy - enemy_xy))
            if d < min_dist:
                min_dist = d
                min_agent = ag

        self._interceptor = min_agent

    # ======================================================================
    # 초기화/이동/시각화 관련
    # ======================================================================
    def _hover(self, name):
        """드론을 완전 정지시키는 호버링 함수"""
        # velocity 0으로 주고 PID 안정화
        self.client.moveByVelocityZAsync(
            vx=0.0, vy=0.0,
            z=self.fixed_z,
            duration=0.3,
            vehicle_name=name
        ).join()

        # AirSim 내부 hover PID 활성화
        try:
            self.client.hoverAsync(vehicle_name=name).join()
        except:
            pass  # 일부 AirSim 버전에서 hoverAsync 없을 수 있음

    def _setup_flight(self):

        # ===== API Control & Arm =====
        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")

        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
            self.client.armDisarm(True, vehicle_name=agent)

        self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
        self.client.armDisarm(True, vehicle_name=self.dynamic_name)

        # ===== Takeoff =====
        cmds = []
        cmds.append(self.client.takeoffAsync(vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.takeoffAsync(vehicle_name=agent))
        cmds.append(self.client.takeoffAsync(vehicle_name=self.dynamic_name))

        for c in cmds:
            c.join()

        # ===== 초기 pose 기록 =====
        self.start_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.start_location[agent] = self.client.simGetObjectPose(agent)
        self.start_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

        # ===== 초기 위치로 이동 =====
        cmds = []

        # 리더
        cmds.append(
            self.client.moveToPositionAsync(
                x=0.0, y=0.0, z=self.fixed_z,
                velocity=5.0,
                vehicle_name="Drone1"
            )
        )

        # 에이전트들
        for agent in self.possible_agents:
            cmds.append(
                self.client.moveToPositionAsync(
                    x=0.0, y=0.0, z=self.fixed_z,
                    velocity=5.0,
                    vehicle_name=agent
                )
            )

        # 동적 장애물
        cmds.append(
            self.client.moveToPositionAsync(
                x=0.0, y=0.0, z=self.fixed_z,
                velocity=5.0,
                vehicle_name=self.dynamic_name
            )
        )

        for c in cmds:
            c.join()

        # ===== Hover 안정화 (진짜 중요) =====
        # 이동 후 드론들이 미세하게 흔들리거나 기울어진 상태를 제거
        self._hover("Drone1")
        for agent in self.possible_agents:
            self._hover(agent)
        self._hover(self.dynamic_name)

        # 안정화 대기 (PID alignment)
        time.sleep(1)


    def _update_leader_movement(self):
        """
        유인기에 아무런 명령도 내리지 않고, 그대로 유지, 유인기는 초기 setup시에 해당 고도로 이동한 후, 아무런 명령없이 대기
        """

        # 2. 시각화는 그대로 유지
        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.1:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

    def _visualize_circles(self):
        try:
            leader_pos = self.client.simGetObjectPose("Drone1").position
            center = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val], dtype=float)

            def ring_points(radius, n=36):
                pts = []
                for i in range(n + 1):
                    ang = (i / n) * 2 * np.pi
                    x = center[0] + radius * np.cos(ang)
                    y = center[1] + radius * np.sin(ang)
                    z = center[2]
                    pts.append(airsim.Vector3r(x, y, z))
                return pts

            line_thickness = 20.0
            self.client.simPlotLineStrip(
                ring_points(self.optimal_distance),
                [1, 1, 0, 0.8],
                line_thickness, 0.15, True
            )
            self.client.simPlotLineStrip(
                ring_points(self.far_cutoff),
                [0, 1, 0, 0.8],
                line_thickness, 0.15, True
            )
        except Exception:
            print("시각화 오류 발생")
            pass

    # ======================================================================
    # 보상/종료 관련
    # ======================================================================
    def _formation_reward(self, agent_pos, leader_pos):
        rel = leader_pos - agent_pos
        dist = math.hypot(float(rel[0]), float(rel[1]))
        if dist < 0.5 or dist > 60.0:
            return -5.0
        ideal = 10.0
        sigma = 10.0
        r = 3.0 * math.exp(-((dist - ideal) ** 2) / (2.0 * sigma ** 2)) - 1.0
        return r

    def _guardian_reward(self, agent_pos, leader_pos, dynamic_pos):
        d_lo = np.linalg.norm(leader_pos[:2] - dynamic_pos[:2])
        d_ao = np.linalg.norm(agent_pos[:2] - dynamic_pos[:2])

        ALERT_DIST = 80.0
        if d_lo > ALERT_DIST:
            return 0.0

        if d_ao < d_lo:
            score = (d_lo - d_ao) / max(d_lo, 1e-3)
            return 2.0 * score
        else:
            return -0.5


    def _compute_reward(self, agent):
        """
        롤 기반 + heading 정렬 + 개별 크레딧 버전

        - Interceptor:
            PN(접근, LOS) + 거리 개선 + heading alignment
        - Defender:
            리더와 optimal_distance 근처 유지 (수비)
        """

        eps = 1e-6

        # -------------------------------
        # 0. 리더 / 적 위치
        # -------------------------------
        enemy_pose = self.current_location[self.dynamic_name].position
        enemy_xy = np.array([enemy_pose.x_val, enemy_pose.y_val], dtype=np.float32)

        leader_pose = self.current_location["Drone1"].position
        leader_xy = np.array([leader_pose.x_val, leader_pose.y_val], dtype=np.float32)

        interceptor = self._interceptor
        is_interceptor = (agent == interceptor)

        # -------------------------------
        # 1. 내 상태 / 적 상태
        # -------------------------------
        my_state = self.client.getMultirotorState(vehicle_name=agent)
        my_pos = np.array([
            my_state.kinematics_estimated.position.x_val,
            my_state.kinematics_estimated.position.y_val,
        ], dtype=np.float32)
        my_vel = np.array([
            my_state.kinematics_estimated.linear_velocity.x_val,
            my_state.kinematics_estimated.linear_velocity.y_val,
        ], dtype=np.float32)

        # heading 벡터
        my_yaw = self.heading_state[agent]
        my_heading_vec = np.array([math.cos(my_yaw), math.sin(my_yaw)], dtype=np.float32)

        target_state = self.client.getMultirotorState(vehicle_name=self.dynamic_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val,
        ], dtype=np.float32)
        target_vel = np.array([
            target_state.kinematics_estimated.linear_velocity.x_val,
            target_state.kinematics_estimated.linear_velocity.y_val,
        ], dtype=np.float32)

        R_vec = target_pos - my_pos   # 에이전트 -> 적
        V_vec = target_vel - my_vel
        dist = float(np.linalg.norm(R_vec))

        # -------------------------------
        # 2. Interceptor 보상
        # -------------------------------
        r_close = 0.0
        r_los   = 0.0
        r_dist  = 0.0
        r_align = 0.0

        if is_interceptor and dist > eps:
            # (1) heading alignment: 적을 바라보면 +
            to_target_unit = R_vec / dist
            align_score = float(np.dot(my_heading_vec, to_target_unit))  # [-1, 1]
            if align_score > 0.0:
                r_align = self.W_ALIGN * align_score   # 정면(1)일 때 최대 보상

            # (2) closing speed: 적에게 실제로 접근할수록 +
            closing_speed = -float(np.dot(R_vec, V_vec)) / (dist + eps)
            V_MAX = 1.0
            closing_norm = float(np.clip(closing_speed / V_MAX, -1.0, 1.0))
            if closing_norm > 0.0:
                r_close = self.W_CLOSE * closing_norm

            # (3) LOS-rate: 시선각 변화율이 작을수록 +
            cross = float(R_vec[0] * V_vec[1] - R_vec[1] * V_vec[0])
            los_rate = cross / (dist**2 + eps)
            LAMBDA_DOT_MAX = 10.0
            los_norm = float(np.clip(abs(los_rate) / LAMBDA_DOT_MAX, 0.0, 1.0))
            r_los = self.W_LOS * (1.0 - los_norm)

            # (4) 거리 개선: 이전 스텝보다 가까워졌으면 +
            d_now = dist
            prev_d = self._prev_d_agent_enemy.get(agent, None)
            if prev_d is not None:
                delta = prev_d - d_now  # +면 더 가까워짐
                if delta > 0.0:
                    delta_clipped = float(
                        np.clip(delta, -self.MAX_DIST_DELTA, self.MAX_DIST_DELTA)
                    )
                    r_dist = self.W_DIST * (delta_clipped / self.MAX_DIST_DELTA)

            self._prev_d_agent_enemy[agent] = d_now

        else:
            # defender도 거리 기록은 유지
            self._prev_d_agent_enemy[agent] = dist

        # -------------------------------
        # 3. Defender 보상: 리더 수비
        # -------------------------------
        r_def = 0.0

        if not is_interceptor:
            d_leader = float(np.linalg.norm(my_pos - leader_xy))
            center = self.optimal_distance
            band   = self.DEF_LEADER_BAND

            offset = abs(d_leader - center)
            if offset < band:
                # center일수록 1, band 바깥이면 0
                ratio = 1.0 - (offset / band)
                r_def = self.W_DEF_LEADER * ratio

        # -------------------------------
        # 4. 최종 보상 (step shaping, ≥ 0 위주)
        # -------------------------------
        reward = r_close + r_los + r_dist + r_align + r_def

        return float(reward), False

    
    def _end_episode(self, reward, status):
        """
        에피소드 종료 헬퍼 (충돌/성공/타임아웃 등 이벤트 발생 시)
        """
        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        # 에피소드 단위 지표 (0/1 플래그)
        is_success = 1 if status == "SUCCESS_DISTANCE_AGENT_DYNAMIC" else 0

        # 리더 피격
        is_leader_hit = 1 if status == "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION" else 0

        # 아군 충돌: 에이전트-에이전트, 에이전트-리더 둘 다 포함
        is_ally_collision = 1 if status in (
            "FAIL_DISTANCE_AGENT_AGENT",
            "FAIL_DISTANCE_AGENT_LEADER",
        ) else 0

        for agent in self.possible_agents:
            _obs_list.append(self._get_obs(agent))
            _rewards_list.append(reward)
            _terminations_list.append(True)

            _infos_list.append({
                agent: {
                    "final_status": status,
                    "reward": reward,

                    # 에피소드 지표 (0 또는 1)
                    "episode_success": is_success,
                    "episode_leader_hit": is_leader_hit,
                    "episode_ally_collision": is_ally_collision,
                }
            })
        
        return _obs_list, _rewards_list, _terminations_list, _infos_list


    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self):
        """
        동적 장애물 FSM (Step Count 기반)
        """
        self._obs_step_timer += 1  # 현재 상태에서의 경과 스텝 증가

        # ==========================================
        # STATE: IDLE (대기 상태)
        # ==========================================
        if self._obstacle_state == "IDLE":
            # 호버링 유지 (위치 고정)
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name)

            # [조건 체크] 정해진 대기 스텝(10~30)이 지났는가?
            if self._obs_step_timer >= self._idle_wait_steps:
                print(f"[DynamicObstacle] {self._obs_step_timer} steps passed. IDLE -> ATTACK!")
                self._obstacle_state = "ATTACK"
                self._obs_step_timer = 0  # 타이머 리셋
        
        # ==========================================
        # STATE: ATTACK (추적 상태)
        # ==========================================
        elif self._obstacle_state == "ATTACK":
            # 1. 유인기 방향 계산
            try:
                leader_pos = self.current_location["Drone1"].position
                obs_pos = self.current_location[self.dynamic_name].position
                
                l_vec = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val])
                o_vec = np.array([obs_pos.x_val, obs_pos.y_val, obs_pos.z_val])
                
                diff = l_vec - o_vec
                dist = np.linalg.norm(diff)
                
                if dist > 0.5:
                    direction = diff / dist
                    speed = 5.0  # 공격 속도 (m/s)
                    vel = direction * speed
                    
                    # 2. 속도 명령 전송 (유도탄 처럼 추적)
                    self.client.moveByVelocityAsync(
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]), 
                        duration=0.1, 
                        vehicle_name=self.dynamic_name
                    )
            except Exception as e:
                print(f"Attack Logic Error: {e}")

            # 3. [안전 장치] 너무 오랫동안(예: 200스텝) 못 맞추면 강제 리셋 (무한 추적 방지)
            if self._obs_step_timer > 500:
                 print("[DynamicObstacle] Attack Timeout. Forcing Reset.")
                 self._reset_obstacle_logic()


    def _teleport_obstacle_randomly(self):
        """장애물을 리더 근처 랜덤 위치로 순간이동 시킴"""
        leader_pos = self.client.simGetObjectPose("Drone1").position
        lx, ly = leader_pos.x_val, leader_pos.y_val
        
        # 50m ~ 60m 반경 내 랜덤 위치
        radius = random.uniform(50.0, 60.0)
        angle = random.uniform(0, 2 * math.pi)
        
        tx = lx + radius * math.cos(angle)
        ty = ly + radius * math.sin(angle)
        tz = self.fixed_z

        # 위치 강제 설정
        pose = airsim.Pose(airsim.Vector3r(tx, ty, tz), airsim.Quaternionr(0,0,0,1))
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.dynamic_name)

        
        # 중요: 순간이동 후 이전 속도(관성) 제거
        self.client.moveByVelocityAsync(0,0,0, duration=0.1, vehicle_name=self.dynamic_name).join()

    def _reset_obstacle_logic(self):
        """
        공격 완료/실패 후 호출:
        1. 랜덤 위치로 순간이동
        2. 상태를 IDLE로 변경
        3. 랜덤 대기 시간(10~30 step) 재설정
        """
        self._teleport_obstacle_randomly()
        
        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(10, 30) # 다음 대기 시간 랜덤 설정
        
        print(f"[DynamicObstacle] Reset to IDLE. Waiting for {self._idle_wait_steps} steps.")
    
    def _check_distance_collision(self, name_a, name_b, threshold):
        pa = self.current_location[name_a].position
        pb = self.current_location[name_b].position

        dx = pa.x_val - pb.x_val
        dy = pa.y_val - pb.y_val

        dist = math.sqrt(dx*dx + dy*dy)
        return dist < threshold, dist


    # ======================================================================
    # RL/PettingZoo API
    # ======================================================================
    @property
    def observation_space(self):
        return [self.observation_spaces[a] for a in self.possible_agents]

    @property
    def action_space(self):
        return [self.action_spaces[a] for a in self.possible_agents]

    @property
    def share_observation_space(self):
        return [self.share_observation_spaces for _ in self.possible_agents]

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _get_obs(self, agent):
        # ---------------------------------------------------------
        # 1. 에이전트(나)의 운동 정보 가져오기 (World Frame)
        # ---------------------------------------------------------
        my_state = self.client.getMultirotorState(vehicle_name=agent)
        my_pos = np.array([
            my_state.kinematics_estimated.position.x_val,
            my_state.kinematics_estimated.position.y_val
        ], dtype=np.float32)
        my_vel = np.array([
            my_state.kinematics_estimated.linear_velocity.x_val,
            my_state.kinematics_estimated.linear_velocity.y_val
        ], dtype=np.float32)

        # ---------------------------------------------------------
        # 2. 동적 장애물(적)의 운동 정보 가져오기
        # ---------------------------------------------------------
        target_state = self.client.getMultirotorState(vehicle_name=self.dynamic_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val
        ], dtype=np.float32)
        target_vel = np.array([
            target_state.kinematics_estimated.linear_velocity.x_val,
            target_state.kinematics_estimated.linear_velocity.y_val
        ], dtype=np.float32)

        # ---------------------------------------------------------
        # 3. 유도탄 공학 필수 물리량 계산 (Intercept Physics)
        # ---------------------------------------------------------
        R_vec = target_pos - my_pos     # 상대 위치
        V_vec = target_vel - my_vel     # 상대 속도
        
        dist = float(np.linalg.norm(R_vec))
        epsilon = 1e-6  # 0 나누기 방지

        # (1) Closing Velocity: - (R·V) / |R|
        closing_speed = -float(np.dot(R_vec, V_vec)) / (dist + epsilon)

        # (2) LOS Rate: (R x V) / |R|^2  (2D 외적)
        cross_prod = float(R_vec[0] * V_vec[1] - R_vec[1] * V_vec[0])
        los_rate = cross_prod / (dist**2 + epsilon)

        # ---- 물리적으로 말이 안 되게 튀는 값은 Box 범위로 클리핑 ----
        closing_speed = float(np.clip(closing_speed, -30.0, 30.0))
        los_rate      = float(np.clip(los_rate,      -10.0, 10.0))

        closing_speed_norm = closing_speed / 30.0   # [-1,1]
        los_rate_norm      = los_rate      / 10.0   # [-1,1]

        # ---------------------------------------------------------
        # 4. 관측값 조립
        # ---------------------------------------------------------
        _leader_feats  = []
        _ally_feats    = []
        _dynamic_feats = []

        # 리더: (상대 방위, 거리)
        angle_diff, distance_diff = self._angle_and_distance(agent, "Drone1")
        angle_norm = angle_diff / math.pi          # [-1,1]
        _leader_feats = [angle_norm, distance_diff]

        # 아군들: (상대 방위, 거리) * N
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            a_diff, d_diff = self._angle_and_distance(agent, other)
            a_norm = a_diff / math.pi             # [-1,1]
            _ally_feats.append([a_norm, d_diff])

        # 적: (상대 방위, 거리) + (closing_speed, los_rate)
        dyn_angle, dyn_dist = self._angle_and_distance(agent, self.dynamic_name)
        dyn_angle_norm = dyn_angle / math.pi      # [-1,1]

        _dynamic_feats = [
            dyn_angle_norm,
            dyn_dist,
            closing_speed,         # 접근 속도
            los_rate               # 시선각 변화율
        ]

        # 역할 플래그: 이 에이전트가 interceptor면 1.0, 아니면 0.0
        role_flag = 1.0 if (self._interceptor == agent) else 0.0       

        # [리더(2), 아군*(2), 적(4)] → 총 10차원
        obs = np.concatenate([
            np.array(_leader_feats,  dtype=np.float32).flatten(),
            np.array(_ally_feats,    dtype=np.float32).flatten(),
            np.array(_dynamic_feats, dtype=np.float32).flatten(),
            np.array([role_flag],    dtype=np.float32),
        ]).astype(np.float32)

        return obs


    def _do_action(self, actions):
        """
        heading_state + speed 방식 제어
        -------------------------------------
        actions[i][0] = yaw_rate  [-1, 1] → [-MAX_YAW, MAX_YAW]
        actions[i][1] = speed     [-1, 1] → [0, MAX_SPEED]

        heading_state[agent]은 환경 내부에서 누적 관리.
        """

        actions = np.clip(actions, -1, 1)

        dt = 0.1  # step interval (duration과 동일하게 유지)

        for i, agent in enumerate(self.possible_agents):

            a = actions[i]

            # ===============================
            # 1) yaw_rate 누적 → heading 업데이트
            # ===============================
            yaw_rate = a[0] * self.MAX_YAW
            self.heading_state[agent] += yaw_rate * dt  # 내부 yaw 상태 업데이트

            # heading 정규화
            self.heading_state[agent] = (self.heading_state[agent] + math.pi) % (2 * math.pi) - math.pi

            # ===============================
            # 2) speed 변환
            # ===============================
            speed = (a[1] + 1) / 2 * self.MAX_SPEED  # [0, MAX_SPEED]

            # ===============================
            # 3) heading 기반 velocity 계산
            # ===============================
            yaw = self.heading_state[agent]

            vx = math.cos(yaw) * speed
            vy = math.sin(yaw) * speed

            # ===============================
            # 4) velocity 명령 전송 (고도 유지)
            # ===============================
            self.client.moveByVelocityZAsync(
                vx=vx,
                vy=vy,
                z=self.fixed_z,
                duration=dt,
                vehicle_name=agent
            )

    
    def _get_rewards(self, per_agent_results):
        # 수정: 각 에이전트가 자기 step 보상을 그대로 받음
        return per_agent_results

    def reset(self, seed=None, options=None):

        self.episode_count += 1
        print(f"에피소드: {self.episode_count} | 소비한 스텝 수: {self.step_count}")

        self.step_count = 0
        self.agents = self.possible_agents[:]

        # 내부 heading 상태 초기화 (그냥 dict만 만들어두고 나중에 적 방향으로 세팅)
        self.heading_state = {a: 0.0 for a in self.possible_agents}

        # ---- AirSim 월드 리셋 및 초기 비행 세팅 ----
        self.client.reset()
        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        # 동적 장애물 초기화 (리더 근처 랜덤 위치 등)
        self._reset_obstacle_logic()

        # 각 드론 현재 위치/자세 갱신
        self._get_current_location()

        # ---- heading_state 를 "현재 적 방향"으로 세팅 ----
        enemy_pos = self.current_location[self.dynamic_name].position
        ex, ey = enemy_pos.x_val, enemy_pos.y_val

        self.heading_state = {}
        for a in self.possible_agents:
            agent_pos = self.current_location[a].position
            ax, ay = agent_pos.x_val, agent_pos.y_val

            dx = ex - ax
            dy = ey - ay

            desired_yaw = math.atan2(dy, dx)  # 에이전트 → 적 방향 (world yaw)
            # [-pi, pi] 범위로 정규화 (사실 atan2가 이미 이렇게 주지만 안전하게)
            desired_yaw = (desired_yaw + math.pi) % (2 * math.pi) - math.pi

            self.heading_state[a] = desired_yaw

        self._update_roles()

        # 이전 포즈/시간/액션 기록 초기화
        self._last_pose.clear()
        self._last_time.clear()
        self._last_action = {
            a: np.zeros(2, dtype=np.float32) for a in self.possible_agents
        }
        self.leader_stop = False

        # PN 보상용 이전 값들 초기화
        self._prev_d_leader_enemy = None
        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._prev_los_angle = {a: None for a in self.possible_agents}

        # 초기 관측
        obs_list = [self._get_obs(a) for a in self.agents]

        print("reset.")

        return obs_list



    def step(self, actions):

        # ===== 스텝 시 초기화 인스턴스들 ===== 
        # - 스텝 카운드 +1
        self.step_count += 1
        # - 에이전트 별 | 관측 / 보상 / 정보 | 인덱스 초기화
        per_agent_obs, per_agent_results, per_agent_infos = [], [], []


        # ===== Action Step =====
        # - 에이전트 액션 적용 (에이전트별 action 구현)
        self._do_action(actions)

        # - 유인기/장애물 이동
        self._update_leader_movement()        
        self._update_dynamic_obstacle()

        # 현재 위치 값 받아오기 (World Frame)
        self._get_current_location()
        #print(self.current_location["Follower1"].orientation)
        self._update_roles()
        # ===== Check Termination Step =====
        for agent in self.possible_agents:
            
            # 이번 스텝에 활용할 거리 정보 데이터 미리 연산 해두기 (각 에이전트 별 기준)
            other_agents = [a for a in self.possible_agents if a != agent]  # 본인이 아닌 다른 에이전트 배열 불러오기  

            ## - 유인기와의 거리
            _distance_leader = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location["Drone1"].position.x_val,
                self.current_location[agent].position.y_val - self.current_location["Drone1"].position.y_val,
                self.current_location[agent].position.z_val - self.current_location["Drone1"].position.z_val
            ])
            ## - 다른 에이전트들 간의 거리 (List)
            _distance_other = [
                np.linalg.norm([
                    self.current_location[agent].position.x_val - self.current_location[other].position.x_val,
                    self.current_location[agent].position.y_val - self.current_location[other].position.y_val,
                    self.current_location[agent].position.z_val - self.current_location[other].position.z_val
                ]) for other in other_agents
            ]
            ## - 동적 장애물과의 거리
            _distance_dynamic = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location[self.dynamic_name].position.x_val,
                self.current_location[agent].position.y_val - self.current_location[self.dynamic_name].position.y_val,
                self.current_location[agent].position.z_val - self.current_location[self.dynamic_name].position.z_val
            ])

            # - 만약 에이전트가 유인기의 범위를 벗어났을 경우,
            if _distance_leader > self.far_cutoff:
                print(
                    f"[이탈] {agent}가 리더와의 거리({_distance_leader:.2f}m)로, "
                    f"이탈 임계값({self.far_cutoff}m) 초과! → 전체 실패(경계 이탈)"
                )
                return self._end_episode(self.REWARD_AGENT_CRASH,"FAIL_AGENT_FAR_CUTOFF")

            
            # - 만약 에이전트가 유인기와 충돌했을 경우,
            hit, d = self._check_distance_collision(agent, "Drone1", threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"⚠️💔[거리-충돌] {agent} ↔ Drone1  (d={d:.2f}) → 전체 실패")
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_LEADER")
            
            # - 만약 에이전트가 에이전트와 충돌했을 경우,
            for other in other_agents:
                hit, d = self._check_distance_collision(agent, other, threshold=self.COLLISION_THRESHOLD)
                if hit:
                    print(f"💥🤖[거리-충돌] {agent} ↔ {other}  (d={d:.2f}) → 전체 실패")
                    return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_AGENT")
            
            # - 만약 유인기가 동적장애물과 충돌했을 경우,
            collisionInfo = self.client.simGetCollisionInfo("Drone1")
            if collisionInfo.has_collided and collisionInfo.object_name == self.dynamic_name:
                print(f"💥[충돌] 유인기가 {collisionInfo.object_name}와 충돌로 → 전체 실패")
                return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION")

            # - 만약 에이전트가 동적 장애물과 충돌했을 경우 (요격 성공),
            hit, d = self._check_distance_collision(agent, self.dynamic_name, threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"🎯🔥[거리-충돌] {agent} ↔ {self.dynamic_name}  (d={d:.2f}) → 요격 성공")
                return self._end_episode(self.REWARD_HIT_ENEMY, "SUCCESS_DISTANCE_AGENT_DYNAMIC")


            # - 종료 조건을 만족하지 못한 경우,
            per_agent_obs.append(self._get_obs(agent))

            _reward, _ = self._compute_reward(agent)
            per_agent_results.append(_reward)
            per_agent_infos.append([f"reward: {_reward}"])

        # 도중에 종료 안되면 다 종료 안함.
        termination_list = [False for _ in self.possible_agents]

        # ===== Rewards Step =====
        rewards_list = self._get_rewards(per_agent_results)

        
        # ===== Observations Step =====
        obs_list = per_agent_obs


        # ===== Infos Step =====
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list