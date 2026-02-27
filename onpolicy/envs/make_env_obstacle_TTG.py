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
        self._timestep = float(dt)

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
        
        # closing_speed_norm, los_rate_norm 범위
        low_vel = -1.0
        high_vel = 1.0
        low_rate = -1.0
        high_rate = 1.0

        # [리더(2)] + [아군(2)*K] + [적(4)*K]
        per_agent_low = (
            [low_bearing, low_dist] +
            [low_bearing, low_dist] * self.num_ally +
            [low_bearing, low_dist, low_vel, low_rate] * self.num_enemy
        )
        per_agent_high = (
            [high_bearing, high_dist] +
            [high_bearing, high_dist] * self.num_ally +
            [high_bearing, high_dist, high_vel, high_rate] * self.num_enemy
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

        # ===== 보상 관련 버퍼/파라미터 =====
        # 에이전트-적 거리의 이전 값 저장 (접근 보상용)
        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}

        # 팀 공통 터미널 보상
        self.REWARD_HIT_ENEMY = 120.0
        self.REWARD_LEADER_HIT = -300.0
        self.REWARD_AGENT_CRASH = -80.0

        # 접근 보상: 이전보다 얼마나 가까워졌는지
        self.W_DIST = 1.0
        self.MAX_DIST_DELTA = 5.0  # 한 스텝당 최대 의미있게 보는 접근량 [m]

        # 리더-적 경쟁(가드) 보상
        self.W_GUARD = 0.5
        self.ALERT_GUARD_DIST = 80.0  # 이 안에 들어오면 방어 관심 구간

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
        dx = float(
            self.current_location[target_drone].position.x_val -
            self.current_location[src_drone].position.x_val
        )
        dy = float(
            self.current_location[target_drone].position.y_val -
            self.current_location[src_drone].position.y_val
        )
        
        # heading_state에서 yaw 사용
        src_yaw = self.heading_state[src_drone]

        distance_diff = math.sqrt(dx**2 + dy**2)

        _angle = math.atan2(dy, dx)

        angle_diff = ((_angle - src_yaw) + math.pi) % (2 * math.pi) - math.pi
    
        return angle_diff, distance_diff

    def _get_current_location(self):
        self.current_location = {}
        self.current_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.current_location[agent] = self.client.simGetObjectPose(agent)
        self.current_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

    # ======================================================================
    # 초기화/이동/시각화 관련
    # ======================================================================
    def _hover(self, name):
        self.client.moveByVelocityZAsync(
            vx=0.0, vy=0.0,
            z=self.fixed_z,
            duration=0.3,
            vehicle_name=name
        ).join()

        try:
            self.client.hoverAsync(vehicle_name=name).join()
        except:
            pass

    def _setup_flight(self):
        # API Control & Arm
        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")

        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
            self.client.armDisarm(True, vehicle_name=agent)

        self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
        self.client.armDisarm(True, vehicle_name=self.dynamic_name)

        # Takeoff
        cmds = []
        cmds.append(self.client.takeoffAsync(vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.takeoffAsync(vehicle_name=agent))
        cmds.append(self.client.takeoffAsync(vehicle_name=self.dynamic_name))

        for c in cmds:
            c.join()

        # 초기 pose 기록
        self.start_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.start_location[agent] = self.client.simGetObjectPose(agent)
        self.start_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

        # 초기 위치로 이동 (리더, 에이전트, 적 모두 원점 주변)
        cmds = []

        cmds.append(
            self.client.moveToPositionAsync(
                x=0.0, y=0.0, z=self.fixed_z,
                velocity=5.0,
                vehicle_name="Drone1"
            )
        )

        for agent in self.possible_agents:
            cmds.append(
                self.client.moveToPositionAsync(
                    x=0.0, y=0.0, z=self.fixed_z,
                    velocity=5.0,
                    vehicle_name=agent
                )
            )

        cmds.append(
            self.client.moveToPositionAsync(
                x=0.0, y=0.0, z=self.fixed_z,
                velocity=5.0,
                vehicle_name=self.dynamic_name
            )
        )

        for c in cmds:
            c.join()

        # Hover 안정화
        self._hover("Drone1")
        for agent in self.possible_agents:
            self._hover(agent)
        self._hover(self.dynamic_name)

        time.sleep(1)

    def _update_leader_movement(self):
        # 리더는 고정, 시각화만
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
    def _compute_reward(self, agent):
        """
        [최종 버전]
        - 헤딩 정렬 보상 제거
        - 롤(interceptor/defender) 제거
        - 두 가지 shaping만 사용:
          1) r_approach: 이전 스텝보다 적과의 거리가 줄어들면 + (접근 보상)
          2) r_guard: 적에 대해 에이전트가 리더보다 더 가깝게 위치하면 +, 뒤에 서 있으면 약한 -
        """

        eps = 1e-6

        # 리더 / 적 위치 (xy만 사용)
        enemy_pose = self.current_location[self.dynamic_name].position
        enemy_xy = np.array([enemy_pose.x_val, enemy_pose.y_val], dtype=np.float32)

        leader_pose = self.current_location["Drone1"].position
        leader_xy = np.array([leader_pose.x_val, leader_pose.y_val], dtype=np.float32)

        # 내 상태
        my_state = self.client.getMultirotorState(vehicle_name=agent)
        my_pos = np.array([
            my_state.kinematics_estimated.position.x_val,
            my_state.kinematics_estimated.position.y_val,
        ], dtype=np.float32)

        # 에이전트-적 거리
        d_agent_enemy = float(np.linalg.norm(my_pos - enemy_xy))

        # -----------------------------
        # 1) 접근 보상 (distance improvement)
        # -----------------------------
        r_approach = 0.0
        prev_d = self._prev_d_agent_enemy.get(agent, None)

        if prev_d is not None:
            delta = prev_d - d_agent_enemy   # +면 이전보다 더 가까움
            if delta > 0.0:
                delta_clipped = float(
                    np.clip(delta, 0.0, self.MAX_DIST_DELTA)
                )
                # [0, 1] 범위로 정규화된 접근량
                r_approach = self.W_DIST * (delta_clipped / self.MAX_DIST_DELTA)

        # 현재 거리를 다음 스텝을 위한 prev 값으로 저장
        self._prev_d_agent_enemy[agent] = d_agent_enemy

        # -----------------------------
        # 2) 리더-적 경쟁(가드) 보상
        #    - 적이 리더 근처(경보 거리) 안에 들어왔을 때만 의미 있음
        #    - 에이전트가 적에 더 가깝게 서 있으면 방어벽 역할 → +
        #    - 오히려 에이전트가 더 멀면, 리더가 더 노출 → 약한 -
        # -----------------------------
        r_guard = 0.0
        d_leader_enemy = float(np.linalg.norm(leader_xy - enemy_xy))

        if d_leader_enemy < self.ALERT_GUARD_DIST:
            if d_agent_enemy < d_leader_enemy:
                # 방어 잘 하는 상황
                diff = d_leader_enemy - d_agent_enemy
                ratio = diff / max(d_leader_enemy, eps)  # [0,1] 근처
                r_guard = self.W_GUARD * ratio
            else:
                # 리더가 더 적에 가까운 상태 → 살짝 패널티
                diff = d_agent_enemy - d_leader_enemy
                ratio = diff / max(d_leader_enemy, eps)
                r_guard = -0.2 * self.W_GUARD * ratio  # 패널티는 살짝만

        reward = r_approach + r_guard

        return float(reward), False

    def _end_episode(self, reward, status):
        """
        에피소드 종료 헬퍼 (충돌/성공/타임아웃 등 이벤트 발생 시)
        - 여기서는 "팀 보상" 개념 유지:
          한 에이전트가 적을 요격하면 모두 REWARD_HIT_ENEMY,
          리더 피격/아군 충돌도 팀 전체에 동일 보상.
        """
        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        is_success = 1 if status == "SUCCESS_DISTANCE_AGENT_DYNAMIC" else 0
        is_leader_hit = 1 if status == "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION" else 0
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
        self._obs_step_timer += 1

        if self._obstacle_state == "IDLE":
            # 제자리 대기
            self.client.moveByVelocityAsync(
                0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name
            )

            if self._obs_step_timer >= self._idle_wait_steps:
                print(f"[DynamicObstacle] {self._obs_step_timer} steps passed. IDLE -> ATTACK!")
                self._obstacle_state = "ATTACK"
                self._obs_step_timer = 0
        
        elif self._obstacle_state == "ATTACK":
            try:
                leader_pos = self.current_location["Drone1"].position
                obs_pos = self.current_location[self.dynamic_name].position
                
                l_vec = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val])
                o_vec = np.array([obs_pos.x_val, obs_pos.y_val, obs_pos.z_val])
                
                diff = l_vec - o_vec
                dist = np.linalg.norm(diff)
                
                if dist > 0.5:
                    direction = diff / dist
                    speed = 4.0  # 공격 속도 (m/s)
                    vel = direction * speed
                    
                    self.client.moveByVelocityAsync(
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                        duration=0.1,
                        vehicle_name=self.dynamic_name
                    )
            except Exception as e:
                print(f"Attack Logic Error: {e}")

            if self._obs_step_timer > 500:
                print("[DynamicObstacle] Attack Timeout. Forcing Reset.")
                self._reset_obstacle_logic()

    def _teleport_obstacle_randomly(self):
        """장애물을 리더 근처 랜덤 위치로 순간이동 시킴"""
        leader_pos = self.client.simGetObjectPose("Drone1").position
        lx, ly = leader_pos.x_val, leader_pos.y_val
        
        radius = random.uniform(55.0, 60.0)
        angle = random.uniform(0, 2 * math.pi)
        
        tx = lx + radius * math.cos(angle)
        ty = ly + radius * math.sin(angle)
        tz = self.fixed_z

        pose = airsim.Pose(
            airsim.Vector3r(tx, ty, tz),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.dynamic_name)

        self.client.moveByVelocityAsync(
            0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name
        ).join()

    def _reset_obstacle_logic(self):
        self._teleport_obstacle_randomly()
        
        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(10, 30)
        
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
        # 1. 내 상태
        my_state = self.client.getMultirotorState(vehicle_name=agent)
        my_pos = np.array([
            my_state.kinematics_estimated.position.x_val,
            my_state.kinematics_estimated.position.y_val
        ], dtype=np.float32)
        my_vel = np.array([
            my_state.kinematics_estimated.linear_velocity.x_val,
            my_state.kinematics_estimated.linear_velocity.y_val
        ], dtype=np.float32)

        # 2. 적 상태
        target_state = self.client.getMultirotorState(vehicle_name=self.dynamic_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val
        ], dtype=np.float32)
        target_vel = np.array([
            target_state.kinematics_estimated.linear_velocity.x_val,
            target_state.kinematics_estimated.linear_velocity.y_val
        ], dtype=np.float32)

        # 3. Intercept 물리량
        R_vec = target_pos - my_pos
        V_vec = target_vel - my_vel
        
        dist = float(np.linalg.norm(R_vec))
        epsilon = 1e-6

        closing_speed = -float(np.dot(R_vec, V_vec)) / (dist + epsilon)
        cross_prod = float(R_vec[0] * V_vec[1] - R_vec[1] * V_vec[0])
        los_rate = cross_prod / (dist**2 + epsilon)

        closing_speed = float(np.clip(closing_speed, -30.0, 30.0))
        los_rate      = float(np.clip(los_rate,      -10.0, 10.0))

        closing_speed_norm = closing_speed / 30.0   # [-1,1]
        los_rate_norm      = los_rate      / 10.0   # [-1,1]

        # 4. 관측 조립
        _leader_feats  = []
        _ally_feats    = []
        _dynamic_feats = []

        # 리더: (상대 방위, 거리) -> 각도 [-1,1]로 정규화
        angle_diff, distance_diff = self._angle_and_distance(agent, "Drone1")
        angle_norm = angle_diff / math.pi
        _leader_feats = [angle_norm, distance_diff]

        # 아군: (상대 방위, 거리)
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            a_diff, d_diff = self._angle_and_distance(agent, other)
            a_norm = a_diff / math.pi
            _ally_feats.append([a_norm, d_diff])

        # 적: (상대 방위, 거리, closing_speed_norm, los_rate_norm)
        dyn_angle, dyn_dist = self._angle_and_distance(agent, self.dynamic_name)
        dyn_angle_norm = dyn_angle / math.pi

        _dynamic_feats = [
            dyn_angle_norm,
            dyn_dist,
            closing_speed_norm,
            los_rate_norm
        ]

        obs = np.concatenate([
            np.array(_leader_feats,  dtype=np.float32).flatten(),
            np.array(_ally_feats,    dtype=np.float32).flatten(),
            np.array(_dynamic_feats, dtype=np.float32).flatten(),
        ]).astype(np.float32)

        return obs

    def _do_action(self, actions):
        """
        heading_state + speed 제어
        actions[i][0] = yaw_rate  [-1,1] -> [-MAX_YAW, MAX_YAW]
        actions[i][1] = speed     [-1,1] -> [0, MAX_SPEED]
        """

        actions = np.clip(actions, -1, 1)
        dt = 0.1

        for i, agent in enumerate(self.possible_agents):
            a = actions[i]

            # yaw 업데이트
            yaw_rate = a[0] * self.MAX_YAW
            self.heading_state[agent] += yaw_rate * dt
            self.heading_state[agent] = (self.heading_state[agent] + math.pi) % (2 * math.pi) - math.pi

            # speed 변환
            speed = (a[1] + 1) / 2 * self.MAX_SPEED

            yaw = self.heading_state[agent]
            vx = math.cos(yaw) * speed
            vy = math.sin(yaw) * speed

            self.client.moveByVelocityZAsync(
                vx=vx,
                vy=vy,
                z=self.fixed_z,
                duration=dt,
                vehicle_name=agent
            )

    def _get_rewards(self, per_agent_results):
        # 각 에이전트 step 보상을 그대로 반환
        return per_agent_results

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        print(f"에피소드: {self.episode_count} | 소비한 스텝 수: {self.step_count}")

        self.step_count = 0
        self.agents = self.possible_agents[:]

        # heading_state 초기화
        self.heading_state = {a: 0.0 for a in self.possible_agents}

        # 월드 리셋 및 초기 비행 세팅
        self.client.reset()
        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        # 동적 장애물 초기화
        self._reset_obstacle_logic()

        # 현재 위치 갱신
        self._get_current_location()

        # heading_state를 "현재 적 방향"으로 세팅
        enemy_pos = self.current_location[self.dynamic_name].position
        ex, ey = enemy_pos.x_val, enemy_pos.y_val

        self.heading_state = {}
        for a in self.possible_agents:
            agent_pos = self.current_location[a].position
            ax, ay = agent_pos.x_val, agent_pos.y_val

            dx = ex - ax
            dy = ey - ay

            desired_yaw = math.atan2(dy, dx)
            desired_yaw = (desired_yaw + math.pi) % (2 * math.pi) - math.pi

            self.heading_state[a] = desired_yaw

        # 이전 거리 버퍼 초기화
        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}

        # 기타 버퍼 초기화
        self._last_pose.clear()
        self._last_time.clear()
        self._last_action = {
            a: np.zeros(2, dtype=np.float32) for a in self.possible_agents
        }
        self.leader_stop = False

        # 에피소드 시작 관측
        obs_list = [self._get_obs(a) for a in self.agents]

        print("reset.")

        return obs_list

    def step(self, actions):
        self.step_count += 1

        per_agent_obs, per_agent_results, per_agent_infos = [], [], []

        # 1) 액션 적용
        self._do_action(actions)

        # 2) 리더/장애물 업데이트
        self._update_leader_movement()
        self._update_dynamic_obstacle()

        # 3) 위치 갱신
        self._get_current_location()

        # 4) 종료조건 체크 + 보상 계산
        for agent in self.possible_agents:
            other_agents = [a for a in self.possible_agents if a != agent]

            # 리더와 거리
            _distance_leader = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location["Drone1"].position.x_val,
                self.current_location[agent].position.y_val - self.current_location["Drone1"].position.y_val,
                self.current_location[agent].position.z_val - self.current_location["Drone1"].position.z_val
            ])

            # 동적 장애물과의 거리
            _distance_dynamic = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location[self.dynamic_name].position.x_val,
                self.current_location[agent].position.y_val - self.current_location[self.dynamic_name].position.y_val,
                self.current_location[agent].position.z_val - self.current_location[self.dynamic_name].position.z_val
            ])

            # 4-1) 범위 이탈
            # if _distance_leader > self.far_cutoff:
            #     print(
            #         f"[이탈] {agent}가 리더와의 거리({_distance_leader:.2f}m)로, "
            #         f"이탈 임계값({self.far_cutoff}m) 초과! → 전체 실패(경계 이탈)"
            #     )
            #     return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_AGENT_FAR_CUTOFF")

            # 4-2) 에이전트-리더 충돌
            hit, d = self._check_distance_collision(agent, "Drone1", threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"⚠️💔[거리-충돌] {agent} ↔ Drone1  (d={d:.2f}) → 전체 실패")
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_LEADER")

            # 4-3) 에이전트-에이전트 충돌
            for other in other_agents:
                hit, d = self._check_distance_collision(agent, other, threshold=self.COLLISION_THRESHOLD)
                if hit:
                    print(f"💥🤖[거리-충돌] {agent} ↔ {other}  (d={d:.2f}) → 전체 실패")
                    return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_AGENT")

            # 4-4) 리더-적 충돌
            collisionInfo = self.client.simGetCollisionInfo("Drone1")
            if collisionInfo.has_collided and collisionInfo.object_name == self.dynamic_name:
                print(f"💥[충돌] 유인기가 {collisionInfo.object_name}와 충돌로 → 전체 실패")
                return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION")

            # 4-5) 에이전트-적 충돌 (요격 성공)
            hit, d = self._check_distance_collision(agent, self.dynamic_name, threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"🎯🔥[거리-충돌] {agent} ↔ {self.dynamic_name}  (d={d:.2f}) → 요격 성공")
                return self._end_episode(self.REWARD_HIT_ENEMY, "SUCCESS_DISTANCE_AGENT_DYNAMIC")

            # 종료 안 되었으면 관측/보상 계산
            per_agent_obs.append(self._get_obs(agent))

            _reward, _ = self._compute_reward(agent)
            per_agent_results.append(_reward)
            per_agent_infos.append([f"reward: {_reward}"])

        termination_list = [False for _ in self.possible_agents]
        rewards_list = self._get_rewards(per_agent_results)
        obs_list = per_agent_obs
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list
