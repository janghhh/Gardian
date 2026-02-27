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
        self.COLLISION_THRESHOLD = 1.5
        self.STOP_DISTANCE_LEADER_OBSTACLE = 1.0

        # 속도/액션 버퍼
        self.vmax_self = 2.0
        self._timestep = float(dt)

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}
        self._last_time = {}

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
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
        # Lidar 메모리: { "AgentName": { "TargetName": [dx, dy] } }
        self.lidar_memory = {}
        self.prev_lidar_pos = {}
        # [추가] Lidar 센서 이름 매핑 (settings.json의 이름과 일치해야 함)
        self.lidar_names = {
            agent: f"{agent}_LidarSensor" for agent in self.possible_agents
        }

        # States
        self.step_count = 0
        self.episode_count = 0

        # 통계 계산을 위한 Deque
        self.stats_history = {
            "win": deque(maxlen=20),
            "coll_leader": deque(maxlen=20),
            "coll_drone": deque(maxlen=20),
            "coll_obj": deque(maxlen=20)
        }

        # 동적 장애물 (초기값)
        self.dynamic_name = "DynamicObstacle" 
        self.enemy_names = [] # reset 시 자동 스캔

        # 동적 장애물 관련
        self.isIdle = None
        self.D_O_STATE = {0: "idle", 1: "attack"}

        # ===== obs / act / share_obs spaces =====
        self.K_ally = 5   
        self.K_enemy = 5
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy

        # ---- [수정] 관측 공간 범위 정의 (상대 좌표 도입) ----
        # 기존: Bearing(-1~1), Dist(0~200)
        # 변경: Rel_X(-1~1), Rel_Y(-1~1) 로 변경 (100m 기준 정규화)
        
        low_rel_pos = -1.0;        high_rel_pos = 1.0       
        low_vel = -1.0; high_vel = 1.0
        low_rate = -1.0; high_rate = 1.0
        low_self_state = -1.0; high_self_state = 1.0

        # [리더(2)] + [아군(2)*K] + [적(4)*K] + [self_state(2)]
        # 적(Enemy)의 경우: [rel_x, rel_y, closing_speed, los_rate]
        per_agent_low = (
            [low_rel_pos, low_rel_pos] +                          # Leader (dx, dy)
            [low_rel_pos, low_rel_pos] * self.num_ally +          # Allies (dx, dy)
            [low_rel_pos, low_rel_pos, low_vel, low_rate] * self.num_enemy + # Enemy
            [low_self_state] * 2                                  # Self (vx, vy)
        )
        per_agent_high = (
            [high_rel_pos, high_rel_pos] +
            [high_rel_pos, high_rel_pos] * self.num_ally +
            [high_rel_pos, high_rel_pos, high_vel, high_rate] * self.num_enemy +
            [high_self_state] * 2
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

        # PN 보상용 버퍼들
        self._prev_d_leader_enemy = None
        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._prev_los_angle = {a: None for a in self.possible_agents}

        # PN-style reward parameters
        self.REWARD_HIT_ENEMY = 100.0   # 성공 보상은 크게
        self.REWARD_LEADER_HIT = -250.0
        self.REWARD_AGENT_CRASH = -500.0 # [중요] 자살 방지 위해 충돌 패널티 대폭 강화

        self.W_CLOSE = 0.5
        self.W_LOS = 0.5
        self.W_DIST = 0.5
        self.W_ALLY = 0.5
        self.STEP_PENALTY = 0.0 # [중요] 시간 패널티 제거 (오래 살아남아 요격하도록)

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

    def _get_lidar_measurement(self, agent_name, target_name):
        """
        Lidar로 타겟을 관측합니다.
        - 감지 성공: 포인트 클라우드 평균 위치 반환 + 메모리 갱신
        - 감지 실패: 메모리에 저장된 '마지막 위치' 반환
        """
        lidar_name = self.lidar_names[agent_name]
        lidar_data = self.client.getLidarData(lidar_name, vehicle_name=agent_name)
        
        # 1. 포인트 데이터 파싱
        # SensorLocalFrame 사용 시: points[:, 0]=전방(x), points[:, 1]=우측(y)
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        
        detected = False
        meas_dx, meas_dy = 0.0, 0.0

        if len(points) > 2:
            # 2. 객체 인식 시뮬레이션 (GT 위치 근처의 포인트만 필터링)
            # 현재 Yaw가 0으로 고정된 시뮬레이션이므로, World Frame 차이 ≈ Body Frame 차이로 근사 가능
            # (만약 드론이 회전한다면 회전 행렬 적용 필요)
            
            my_pos = self.current_location[agent_name].position
            tgt_pos = self.current_location[target_name].position
            
            gt_dx = tgt_pos.x_val - my_pos.x_val
            gt_dy = tgt_pos.y_val - my_pos.y_val
            
            # Lidar 포인트들 중 실제 타겟 위치 반경 3m 내에 있는 점만 추출
            # (SensorLocalFrame이므로 points 자체가 상대좌표임)
            dist_sq = (points[:, 0] - gt_dx)**2 + (points[:, 1] - gt_dy)**2
            mask = dist_sq < (3.0)**2  
            target_points = points[mask]

            if len(target_points) > 0:
                # 감지 성공! 포인트들의 평균을 측정값으로 사용
                mean_pos = np.mean(target_points, axis=0)
                raw_dx = mean_pos[0]
                raw_dy = mean_pos[1]
                
                # 정규화 (-1 ~ 1, 100m 기준)
                norm_dx = np.clip(raw_dx / 100.0, -1.0, 1.0)
                norm_dy = np.clip(raw_dy / 100.0, -1.0, 1.0)
                
                # 메모리 갱신
                self.lidar_memory[agent_name][target_name] = [float(norm_dx), float(norm_dy)]
                detected = True
        
        # 3. 반환 (감지했으면 갱신된 값, 못했으면 메모리 값)
        final_val = self.lidar_memory[agent_name][target_name]
        
        # 디버깅용 (필요 시 주석 해제)
        # if not detected:
        #    print(f"[{agent_name}] Lost {target_name}! Using Memory: {final_val}")
            
        return final_val[0], final_val[1], detected
    def _calculate_lidar_dynamics(self, agent, target, curr_x, curr_y):
        """
        [핵심] GT 없이 오직 Lidar의 (현재 위치 - 이전 위치) 차분으로 속도 정보 계산
        """
        # 1. 이전 위치 가져오기 (Reset 시 초기화 필수)
        if agent not in self.prev_lidar_pos: self.prev_lidar_pos[agent] = {}
        if target not in self.prev_lidar_pos[agent]: self.prev_lidar_pos[agent][target] = [curr_x, curr_y]
        
        prev_x, prev_y = self.prev_lidar_pos[agent][target]
        
        # 2. dt 체크 (0 방지)
        dt = self.dt if self.dt > 1e-6 else 0.01

        # 3. 상대 속도 벡터 추정 (Relative Velocity)
        # V_rel = (P_curr - P_prev) / dt
        vx_rel = (curr_x - prev_x) / dt
        vy_rel = (curr_y - prev_y) / dt

        # 4. 거리 및 벡터 연산
        R_vec = np.array([curr_x, curr_y])   # 상대 위치
        V_vec = np.array([vx_rel, vy_rel])   # 상대 속도 (추정)
        dist = np.linalg.norm(R_vec) + 1e-6

        # 5. PN 유도 변수 계산
        # Closing Speed: 가까워지면 +, 멀어지면 -
        closing_speed = -float(np.dot(R_vec, V_vec)) / dist
        
        # LOS Rate: 시선각 변화율
        cross_prod = float(R_vec[0]*V_vec[1] - R_vec[1]*V_vec[0])
        los_rate = cross_prod / (dist**2)
        
        # 6. 정규화 (학습용)
        # 30m/s, 10rad/s는 경험적 Max 값
        norm_closing = np.clip(closing_speed / 30.0, -1.0, 1.0)
        norm_los = np.clip(los_rate / 10.0, -1.0, 1.0)

        return norm_closing, norm_los
    
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
        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")

        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
            self.client.armDisarm(True, vehicle_name=agent)

        self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
        self.client.armDisarm(True, vehicle_name=self.dynamic_name)

        # 1. 이륙 (Takeoff) - 시동 걸기용
        cmds = []
        cmds.append(self.client.takeoffAsync(vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.takeoffAsync(vehicle_name=agent))
        cmds.append(self.client.takeoffAsync(vehicle_name=self.dynamic_name))

        for c in cmds:
            c.join()

        # 2. [수정] 이동(MoveTo) 대신 텔레포트(SetPose) 사용
        # 물리적 이동 없이 좌표를 강제로 찍어버려서 관성을 0으로 만듭니다.
        
        # (1) Leader 위치 고정
        pose_leader = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose_leader, True, vehicle_name="Drone1")
        
        # (2) Followers 위치 고정 (겹치지 않게 약간 분산시켜도 됨, 여기선 0,0으로 모음)
        # 만약 에이전트끼리 겹쳐서 튕겨나간다면 x, y에 약간의 오프셋을 주어야 합니다.
        for i, agent in enumerate(self.possible_agents):
            # 예: 에이전트들을 리더 주변에 약간 띄워서 배치하고 싶다면 아래 주석 해제
            # offset_x = (i % 2) * 2.0 - 1.0  
            # offset_y = (i // 2) * 2.0 - 1.0
            # pose_agent = airsim.Pose(airsim.Vector3r(offset_x, offset_y, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
            
            # 현재는 모두 0,0에 겹쳐도 물리 충돌 무시 설정이 되어있다면 괜찮지만, 보통은 띄우는게 좋습니다.
            # 일단 기존 코드 의도대로 0,0에 둡니다.
            pose_agent = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
            self.client.simSetVehiclePose(pose_agent, True, vehicle_name=agent)

        # (3) 적(DynamicObstacle) 위치 고정 (초기화는 나중에 _reset_obstacle_logic에서 하겠지만 안전상 고정)
        pose_enemy = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose_enemy, True, vehicle_name=self.dynamic_name)

        # 3. [중요] 속도 0으로 강제 초기화 (Momentum Kill)
        # 위치를 옮겨도 이전 속도 벡터가 남아있을 수 있으므로 0으로 덮어씁니다.
        cmds = []
        cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=agent))
        cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name))
        
        for c in cmds:
            c.join()
            
        time.sleep(0.5) # 물리 엔진 안정화 대기

        # 4. 초기 위치 저장
        self.start_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.start_location[agent] = self.client.simGetObjectPose(agent)
        self.start_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

        # Hover 함수 호출 (최종 안정화)
        self._hover("Drone1")
        for agent in self.possible_agents:
            self._hover(agent)
        self._hover(self.dynamic_name)

    def _update_leader_movement(self):
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
            pass

    # ======================================================================
    # 보상/종료 관련
    # ======================================================================
    def _compute_reward(self, agent):
        # 타겟: 0번 적 (또는 가장 가까운 적)
        # 만약 적이 없으면 리더를 타겟으로 잡음 (방어코드)
        target = self.enemy_names[0] if self.enemy_names else "Drone1"
        
        # 1. 정규화된 좌표 가져오기
        norm_pos = self.lidar_memory[agent][target]
        
        # [수정 1] 물리량 계산을 위해 미터(meter) 단위로 복원 (x100.0)
        curr_x_meter = norm_pos[0] * 100.0
        curr_y_meter = norm_pos[1] * 100.0
        
        # 2. Lidar 기반 동역학(속도) 계산
        c_speed, l_rate = self._calculate_lidar_dynamics(agent, target, curr_x_meter, curr_y_meter)
        
        # 3. 보상 계산
        r_close = self.W_CLOSE * c_speed if c_speed > 0 else 0.0
        r_los = self.W_LOS * (1.0 - abs(l_rate))
        
        return float(r_close + r_los - self.STEP_PENALTY)

    def _end_episode(self, reward, status):
        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        # [수정 1] 성공/실패 판정 로직을 TTG와 동일하게 상세화
        is_success = 1.0 if status == "SUCCESS_DISTANCE_AGENT_DYNAMIC" else 0.0
        
        # 리더 피격 (적 -> 리더)
        is_leader_hit = 1.0 if "LEADER_AND_DYNAMIC" in status else 0.0
        
        # 아군 충돌 (에이전트끼리 OR 에이전트가 리더와 충돌)
        # PN 기존 로직은 "AGENT_AGENT"만 체크했으나, 리더와의 충돌도 아군 충돌로 보는 것이 맞음
        if "AGENT_AGENT" in status or "AGENT_LEADER" in status:
            is_ally_collision = 1.0 
        else:
            is_ally_collision = 0.0
            
        # 이탈 (Far Cutoff)
        is_obj_collision = 1.0 if "FAR_CUTOFF" in status else 0.0

        self.stats_history["win"].append(is_success)
        self.stats_history["coll_leader"].append(is_leader_hit)
        self.stats_history["coll_drone"].append(is_ally_collision)
        self.stats_history["coll_obj"].append(is_obj_collision)

        def get_rate(key):
            if len(self.stats_history[key]) > 0:
                return sum(self.stats_history[key]) / len(self.stats_history[key])
            return 0.0

        win_rate = get_rate("win")

        for agent in self.possible_agents:
            _obs_list.append(self._get_obs(agent))
            _rewards_list.append(reward)
            _terminations_list.append(True)

            # [수정 2] info 구조를 { "AgentName": { ... } } 형태로 중첩 (TTG와 통일)
            # Runner가 이 구조를 인식하여 wandb 로그를 수집합니다.
            _infos_list.append({
                agent: {
                    "final_status": status,
                    "episode_success": is_success,
                    "episode_leader_hit": is_leader_hit,
                    "episode_ally_collision": is_ally_collision,
                    "win_rate": win_rate,
                    "cur_episode_steps": self.step_count
                }
            })
        
        print(f"[{self.episode_count} Ep] WinRate: {win_rate:.2f} | Status: {status}")
        return _obs_list, _rewards_list, _terminations_list, _infos_list
    
    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self):
        self._obs_step_timer += 1
        target_speed = 3.0 

        if self._obstacle_state == "IDLE":
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name)
            if self._obs_step_timer >= self._idle_wait_steps:
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
                    vel = direction * target_speed

                    self.client.moveByVelocityAsync(
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                        duration=0.1,
                        vehicle_name=self.dynamic_name
                    )
            except Exception as e:
                print(f"Attack Logic Error: {e}")

    def _teleport_obstacle_randomly(self):
        leader_pos = self.client.simGetVehiclePose("Drone1").position
        lx, ly = leader_pos.x_val, leader_pos.y_val
        radius = 55.0
        angle = random.uniform(0, 2 * math.pi)

        tx = lx - 20 + radius * math.cos(angle)
        ty = ly - 20 + radius * math.sin(angle)
        tz = self.fixed_z

        pose = airsim.Pose(airsim.Vector3r(tx, ty, tz), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.dynamic_name)
        self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name).join()

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

        dist = math.sqrt(dx * dx + dy * dy)
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

    # ----------------------------------------------------------------
    # [2] 관측 (Observation) - 거리순 정렬 + Padding
    # ----------------------------------------------------------------
    def _get_obs(self, agent):
        # 1. 내 상태 (IMU 속도) - 이건 내 정보니까 GT 써도 됨 (혹은 getImuData 사용)
        try:
            my_state = self.client.getMultirotorState(vehicle_name=agent)
            vx = my_state.kinematics_estimated.linear_velocity.x_val
            vy = my_state.kinematics_estimated.linear_velocity.y_val
            norm_vx = np.clip(vx / self.MAX_SPEED, -1.0, 1.0)
            norm_vy = np.clip(vy / self.MAX_SPEED, -1.0, 1.0)
        except:
            norm_vx, norm_vy = 0.0, 0.0

        # 2. 리더 (고정 1개)
        lx, ly, _ = self._get_lidar_measurement(agent, "Drone1")
        # 정규화 (100m 기준)
        _leader_feats = [np.clip(lx/100.0, -1, 1), np.clip(ly/100.0, -1, 1)]

        # 3. 아군 (KNN + Padding)
        other_agents = [a for a in self.possible_agents if a != agent]
        ally_candidates = []
        for other in other_agents:
            ox, oy, _ = self._get_lidar_measurement(agent, other)
            dist_sq = ox**2 + oy**2
            ally_candidates.append({"d": dist_sq, "v": [np.clip(ox/100.0, -1, 1), np.clip(oy/100.0, -1, 1)]})
        
        ally_candidates.sort(key=lambda x: x["d"])
        _ally_feats = []
        for i in range(self.K_ally):
            if i < len(ally_candidates): _ally_feats.extend(ally_candidates[i]["v"])
            else: _ally_feats.extend([0.0, 0.0]) # Padding

        # 4. 적군 (KNN + Padding + Pure Lidar Dynamics)
        enemy_candidates = []
        for e_name in self.enemy_names:
            # (1) Lidar로 위치 측정 (Meter 단위)
            ex, ey, detected = self._get_lidar_measurement(agent, e_name)
            
            # (2) 이전 위치와 비교하여 Closing Speed, LOS Rate 계산 (NO GT!)
            if detected:
                c_speed, l_rate = self._calculate_lidar_dynamics(agent, e_name, ex, ey)
            else:
                # 감지 못했으면 속도 정보 모름 (0 처리)
                c_speed, l_rate = 0.0, 0.0
            
            dist_sq = ex**2 + ey**2
            
            # Feature: [rel_x, rel_y, closing, los] (모두 정규화됨)
            feats = [
                np.clip(ex/100.0, -1, 1), 
                np.clip(ey/100.0, -1, 1), 
                c_speed, 
                l_rate
            ]
            enemy_candidates.append({"d": dist_sq, "v": feats})

        enemy_candidates.sort(key=lambda x: x["d"])
        _enemy_feats = []
        for i in range(self.K_enemy):
            if i < len(enemy_candidates): _enemy_feats.extend(enemy_candidates[i]["v"])
            else: _enemy_feats.extend([0.0, 0.0, 0.0, 0.0]) # Padding

        # 5. 최종 결합
        obs = np.concatenate([
            _leader_feats,
            _ally_feats,
            _enemy_feats,
            [norm_vx, norm_vy]
        ], dtype=np.float32)
        
        return obs
    

    def _do_action(self, actions):
        actions = np.clip(actions, -1, 1)
        dt = self.dt

        for i, agent in enumerate(self.possible_agents):
            a = actions[i]
            # [2D UFO Mode]
            # a[0]: 전후 (Global X), a[1]: 좌우 (Global Y)
            # YawRate: 0 (Fixed)
            
            v_forward = float(a[0]) * self.MAX_SPEED  
            v_lateral = float(a[1]) * self.MAX_SPEED

            sp = math.hypot(v_forward, v_lateral)
            if sp > self.MAX_SPEED:
                s = self.MAX_SPEED / (sp + 1e-6)
                v_forward *= s
                v_lateral *= s

            vx = v_forward 
            vy = v_lateral 

            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0.0)

            self.client.moveByVelocityZAsync(
                vx=vx, vy=vy, z=self.fixed_z, duration=dt,
                yaw_mode=yaw_mode,
                vehicle_name=agent
            )

    def _get_rewards(self, per_agent_results):
        return [np.mean(per_agent_results) for _ in self.possible_agents]

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        print(f"에피소드: {self.episode_count} | 소비한 스텝 수: {self.step_count}")

        self.step_count = 0
        self.agents = self.possible_agents[:]

        self.client.reset()

        # [추가] 시뮬레이터 내의 적군(DynamicObstacle로 시작하는 모든 객체) 자동 감지
        all_vehicles = self.client.listVehicles()
        self.enemy_names = sorted([v for v in all_vehicles if v.startswith("DynamicObstacle")])
        if not self.enemy_names:
            # 적이 하나도 없으면 기본 이름 하나 추가 (패딩으로 처리됨)
            self.enemy_names = ["DynamicObstacle"] 
        
        # 기본 타겟 설정 (보상 계산용, 가장 첫 번째 적을 메인으로 가정하거나 로직 수정 가능)
        self.dynamic_name = self.enemy_names[0]

        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        self._reset_obstacle_logic()
        self._get_current_location()

        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._last_pose.clear()
        self._last_time.clear()

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}

        # 메모리 및 이전 위치 초기화
        self.lidar_memory = {}
        self.prev_lidar_pos = {}
        
        # 메모리 초기화
        for agent in self.possible_agents:
            self.lidar_memory[agent] = {}
            self.prev_lidar_pos[agent] = {}
            # 본인과 리더
            self.lidar_memory[agent]["Drone1"] = [1.0, 1.0]
            self.prev_lidar_pos[agent]["Drone1"] = [100.0, 100.0]
            
            # 모든 잠재적 적군 초기화
            for e_name in self.enemy_names:
                self.lidar_memory[agent][e_name] = [1.0, 1.0]
                self.prev_lidar_pos[agent][e_name] = [100.0, 100.0]

            # 아군들
            for other in self.possible_agents:
                if agent != other:
                    self.lidar_memory[agent][other] = [1.0, 1.0]
                    self.prev_lidar_pos[agent][other] = [100.0, 100.0]

        obs_list = [self._get_obs(a) for a in self.agents]
        return obs_list

    def step(self, actions):
        self.step_count += 1

        if self.step_count >= 300:
            print(f"⏳[시간 초과] 스텝 {self.step_count} 도달! → 실패 처리")
            return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_TIMEOUT_LEADER_HIT")
        
        per_agent_obs, per_agent_results, per_agent_infos = [], [], []

        self._do_action(actions)
        self._update_leader_movement()
        self._update_dynamic_obstacle()
        self._get_current_location() # 충돌 판정용

        for agent in self.possible_agents:
            # 이탈 체크
            _distance_leader = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location["Drone1"].position.x_val,
                self.current_location[agent].position.y_val - self.current_location["Drone1"].position.y_val,
                self.current_location[agent].position.z_val - self.current_location["Drone1"].position.z_val
            ])

            if _distance_leader > self.far_cutoff:
                print(f"[이탈] {agent} 리더 거리 초과! → 전체 실패")
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_AGENT_FAR_CUTOFF")

            # 리더와 충돌
            hit, d = self._check_distance_collision(agent, "Drone1", threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"⚠️💔[충돌] {agent} ↔ Leader")
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_LEADER")

            # 아군끼리 충돌
            other_agents = [a for a in self.possible_agents if a != agent]
            for other in other_agents:
                hit, d = self._check_distance_collision(agent, other, threshold=self.COLLISION_THRESHOLD)
                if hit:
                    print(f"💥🤖[충돌] {agent} ↔ {other}")
                    return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_AGENT")

            # 리더 피격 (Enemy -> Leader)
            collisionInfo = self.client.simGetCollisionInfo("Drone1")
            if collisionInfo.has_collided and collisionInfo.object_name == self.dynamic_name:
                print(f"💥[피격] 리더가 적에게 피격됨!")
                return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION")

            # 요격 성공 (Agent -> Enemy)
            hit, d = self._check_distance_collision(agent, self.dynamic_name, threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"🎯🔥[요격] {agent}가 적을 요격 성공!")
                return self._end_episode(self.REWARD_HIT_ENEMY, "SUCCESS_DISTANCE_AGENT_DYNAMIC")

           # --- [관측 및 보상] ---
            
            # 1. 관측 (여기서 self.lidar_memory가 갱신됨!)
            obs = self._get_obs(agent)
            per_agent_obs.append(obs)

            # 2. 보상 (갱신된 메모리와 이전 위치를 비교하여 계산)
            _reward = self._compute_reward(agent)
            per_agent_results.append(_reward)
            per_agent_infos.append({"step_reward": _reward})
            
            # 3. "모든" 적군에 대해 이전 위치 갱신 (중요!)
            targets_to_update = self.enemy_names + ["Drone1"]
            
            for t_name in targets_to_update:
                if t_name in self.lidar_memory[agent]:
                    norm_curr = self.lidar_memory[agent][t_name]
                    # 정규화 풀어서 저장 (미터 단위)
                    real_curr = [norm_curr[0] * 100.0, norm_curr[1] * 100.0]
                    self.prev_lidar_pos[agent][t_name] = real_curr

        termination_list = [False for _ in self.possible_agents]
        rewards_list = self._get_rewards(per_agent_results)
        obs_list = per_agent_obs
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list