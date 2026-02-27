import gymnasium as gym
import numpy as np
import airsim
import math
import time
import random
from collections import deque


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
        dt=0.01,
        do_visualize=True
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]

        # 충돌 관련 설정
        self.COLLISION_THRESHOLD = 2.0
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
        self.leader_velocity = float(leader_velocity)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)

        # States
        self.step_count = 0
        self.episode_count = 0

        self.stats_history = {
            "win": deque(maxlen=20),            # 요격 성공
            "coll_leader": deque(maxlen=20),    # 리더와 충돌
            "coll_drone": deque(maxlen=20),     # 아군끼리 충돌
            "coll_obj": deque(maxlen=20)        # 이탈/기타 충돌
        }
        self.difficulty_stage = 0

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
        # PN 보상용 버퍼들
        self._prev_d_leader_enemy = None
        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._prev_los_angle = {a: None for a in self.possible_agents}

        # PN-style reward parameters
        self.REWARD_HIT_ENEMY = 100.0
        self.REWARD_LEADER_HIT = -250.0
        self.REWARD_AGENT_CRASH = -50.0

        self.W_CLOSE = 1.0
        self.W_LOS = 1.0
        self.W_DIST = 1.0
        self.W_ALLY = 1.0    

        self.MAX_DIST_DELTA = 5.0
        self.MAX_ANGLE_DELTA = math.radians(20.0)
        self.ALLY_SAFE_DIST = 5.0          
                     

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
        PN 스타일 + 양수 shaping만 사용하는 보상:
        - 적에게 접근(Closing)하면 + (멀어지면 0)
        - LOS-rate가 작을수록 + (심하게 흔들리면 0)
        - 스텝당 명시적인 음수 패널티는 없음
        """

        # -----------------------------------------------------
        # 1. 에이전트 / 적 상태 가져오기 (2D 위치, 속도)
        # -----------------------------------------------------
        my_state = self.client.getMultirotorState(vehicle_name=agent)
        my_pos = np.array([
            my_state.kinematics_estimated.position.x_val,
            my_state.kinematics_estimated.position.y_val,
        ], dtype=np.float32)
        my_vel = np.array([
            my_state.kinematics_estimated.linear_velocity.x_val,
            my_state.kinematics_estimated.linear_velocity.y_val,
        ], dtype=np.float32)

        target_state = self.client.getMultirotorState(vehicle_name=self.dynamic_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val,
        ], dtype=np.float32)
        target_vel = np.array([
            target_state.kinematics_estimated.linear_velocity.x_val,
            target_state.kinematics_estimated.linear_velocity.y_val,
        ], dtype=np.float32)

        # 상대 위치 / 속도
        R_vec = target_pos - my_pos   # (x, y)
        V_vec = target_vel - my_vel   # (vx, vy)

        dist = float(np.linalg.norm(R_vec))
        eps = 1e-6

        # -----------------------------------------------------
        # 2. Closing 속도 Vc 계산  (양수 = 접근, 음수 = 멀어짐)
        #    Vc = - (R·V) / |R|
        # -----------------------------------------------------
        closing_speed = -float(np.dot(R_vec, V_vec)) / (dist + eps)

        # 설계용 최대 closing 속도 스케일 (상황에 맞게 조정)
        V_MAX = 30.0  # m/s 정도 상한 가정

        # 정규화 → [-1, 1] 근사
        closing_norm = closing_speed / V_MAX
        closing_norm = float(np.clip(closing_norm, -1.0, 1.0))

        # 양수일 때(진짜 접근 중일 때)만 보상, 나머지는 0
        if closing_norm > 0.0:
            r_close = self.W_CLOSE * closing_norm
        else:
            r_close = 0.0

        # -----------------------------------------------------
        # 3. LOS-rate (시선각 변화율) 계산
        #    λ̇ = (R_x V_y - R_y V_x) / |R|^2
        # -----------------------------------------------------
        cross = float(R_vec[0] * V_vec[1] - R_vec[1] * V_vec[0])
        los_rate = cross / (dist**2 + eps)  # rad/s 근사

        # 설계용 최대 LOS rate 스케일
        LAMBDA_DOT_MAX = 10.0  # rad/s 정도 상한 가정

        # 얼마나 많이 흔들리는지 [0,1]로 정규화
        los_norm = abs(los_rate) / LAMBDA_DOT_MAX
        los_norm = float(np.clip(los_norm, 0.0, 1.0))

        # 0일수록 좋고, 1에 가까울수록 나쁨 → (1 - los_norm)
        # => 최소 0, 최대 W_LOS
        r_los = self.W_LOS * (1.0 - los_norm)

        # -----------------------------------------------------
        # 4. 스텝 보상 = r_close + r_los (둘 다 ≥ 0)
        # -----------------------------------------------------
        reward = r_close + r_los

        return float(reward), False

    
    def _end_episode(self, reward, status):
        """
        에피소드 종료 처리
        - 통계(승률) 업데이트 (커리큘럼용)
        - 개별 에피소드 결과 플래그 반환 (로깅용)
        """
        # 1. 이번 에피소드 결과 판별 (0 또는 1)
        is_success = 1 if status == "SUCCESS_DISTANCE_AGENT_DYNAMIC" else 0
        is_leader_hit = 1 if "LEADER" in status else 0
        is_ally_collision = 1 if "AGENT_AGENT" in status else 0
        
        # '이탈'은 기타 장애물 충돌로 간주 (필요 시 수정 가능)
        is_obj_collision = 1 if "FAR_CUTOFF" in status else 0

        # 2. Env 내부 메모장(deque)에 기록 (커리큘럼 판단용)
        self.stats_history["win"].append(is_success)
        self.stats_history["coll_leader"].append(is_leader_hit)
        self.stats_history["coll_drone"].append(is_ally_collision)
        self.stats_history["coll_obj"].append(is_obj_collision)

        # 3. 최근 20판 평균 계산 (WandB 그래프용)
        def get_rate(key):
            if len(self.stats_history[key]) > 0:
                return sum(self.stats_history[key]) / len(self.stats_history[key])
            return 0.0

        win_rate = get_rate("win")
        coll_rate_leader = get_rate("coll_leader")
        coll_rate_drone = get_rate("coll_drone")
        coll_rate_obj = get_rate("coll_obj")

        # 4. 반환값 생성
        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        for agent in self.possible_agents:
            _obs_list.append(self._get_obs(agent))
            _rewards_list.append(reward)
            _terminations_list.append(True)
            
            # [수정됨] 기존 키(is_success)와 새로운 키(win_rate)를 모두 포함
            _infos_list.append({
                agent: {
                    "final_status": status,
                    "reward": reward,
                    
                    # === [복구됨] 이번 에피소드의 Raw 결과 (0 or 1) ===
                    "episode_success": is_success,
                    "episode_leader_hit": is_leader_hit,
                    "episode_ally_collision": is_ally_collision,
                    
                    # === [유지됨] 최근 20판 평균 (0.0 ~ 1.0) ===
                    "win_rate": win_rate,
                    "coll_rate_leader": coll_rate_leader,
                    "coll_rate_drone": coll_rate_drone,
                    "coll_rate_obj": coll_rate_obj,
                    
                    # === 난이도 정보 ===
                    "difficulty_stage": self.difficulty_stage,
                    "cur_episode_steps": self.step_count
                }
            })
        
        # 콘솔 로그에는 간단하게 평균과 스테이지 정보 출력
        print(f"[{self.episode_count} Ep] Stage: {self.difficulty_stage} | WinRate: {win_rate:.2f} | Status: {status}")
        
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
                    speed = 5.0  # 공격 속도 (m/s)
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
        leader_pos = self.client.simGetVehiclePose("Drone1").position
        lx, ly = leader_pos.x_val, leader_pos.y_val
        radius = 55.0

        angle = random.uniform(0, 2 * math.pi)
        
        tx = lx-20 + radius * math.cos(angle)
        ty = ly-20 + radius * math.sin(angle)
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

        # ---------------------------------------------------------
        # 4. 관측값 조립
        # ---------------------------------------------------------
        _leader_feats  = []
        _ally_feats    = []
        _dynamic_feats = []

        # 리더: (상대 방위, 거리)
        _leader_feats = [self._angle_and_distance(agent, "Drone1")]

        # 아군들: (상대 방위, 거리) * N
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            _ally_feats.append(self._angle_and_distance(agent, other))

        # 적: (상대 방위, 거리) + (closing_speed, los_rate)
        base_dynamic_feat = self._angle_and_distance(agent, self.dynamic_name)  # (angle, dist)

        _dynamic_feats = [
            base_dynamic_feat[0],  # 상대 방위
            base_dynamic_feat[1],  # 상대 거리
            closing_speed,         # 접근 속도
            los_rate               # 시선각 변화율
        ]

        # [리더(2), 아군*(2), 적(4)] → 총 10차원
        obs = np.concatenate([
            np.array(_leader_feats,  dtype=np.float32).flatten(),
            np.array(_ally_feats,    dtype=np.float32).flatten(),
            np.array(_dynamic_feats, dtype=np.float32).flatten(),
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
                vx=float(vx),       # <--- float()로 감싸기
                vy=float(vy),       # <--- float()로 감싸기
                z=float(self.fixed_z),        # <--- float()로 감싸기
                duration=float(dt), # <--- float()로 감싸기
                vehicle_name=agent
            )

    
    def _get_rewards(self, per_agent_results):
        return [np.mean(per_agent_results) for _ in self.possible_agents]

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
            if _distance_leader > self.far_cutoff:
                print(
                    f"[이탈] {agent}가 리더와의 거리({_distance_leader:.2f}m)로, "
                    f"이탈 임계값({self.far_cutoff}m) 초과! → 전체 실패(경계 이탈)"
                )
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_AGENT_FAR_CUTOFF")

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
