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

        # 동적 장애물 관련
        self.isIdle = None
        self.D_O_STATE = {0: "idle", 1: "attack"}

        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1
        self.K_enemy = 1
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy

        # ---- [수정] 관측 공간 범위 정의 (상대 좌표 도입) ----
        # 기존: Bearing(-1~1), Dist(0~200)
        # 변경: Rel_X(-1~1), Rel_Y(-1~1) 로 변경 (200m 기준 정규화)
        
        low_rel_pos = -1.0
        high_rel_pos = 1.0
        
        low_vel = -1.0
        high_vel = 1.0
        low_rate = -1.0
        high_rate = 1.0
        low_self_state = -1.0
        high_self_state = 1.0

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

        self.dynamic_name = "DynamicObstacle"

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

    # ======================================================================
    # [수정] 헬퍼 메서드: 상대 좌표(dx, dy) 반환으로 변경
    # ======================================================================
    def _get_relative_pos(self, src_drone, target_name):
        """
        [수정] 유클리드 거리 대신 상대 좌표(dx, dy)를 반환합니다.
        World Frame 기준이지만, Yaw가 고정되어 있으므로 에이전트 입장에서도 직관적입니다.
        200m 범위로 정규화(-1 ~ 1)합니다.
        """
        src_pos = self.current_location[src_drone].position
        tgt_pos = self.current_location[target_name].position

        dx = tgt_pos.x_val - src_pos.x_val
        dy = tgt_pos.y_val - src_pos.y_val

        # 200m 맵 크기 기준 정규화 (필요시 far_cutoff 등을 써도 되지만 200m가 넉넉함)
        norm_dx = np.clip(dx / 200.0, -1.0, 1.0)
        norm_dy = np.clip(dy / 200.0, -1.0, 1.0)

        return float(norm_dx), float(norm_dy)

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

        R_vec = target_pos - my_pos
        V_vec = target_vel - my_vel

        dist = float(np.linalg.norm(R_vec))
        eps = 1e-6

        closing_speed = -float(np.dot(R_vec, V_vec)) / (dist + eps)

        V_MAX = 30.0
        closing_norm = closing_speed / V_MAX
        closing_norm = float(np.clip(closing_norm, -1.0, 1.0))

        if closing_norm > 0.0:
            r_close = self.W_CLOSE * closing_norm
        else:
            r_close = 0.0

        cross = float(R_vec[0] * V_vec[1] - R_vec[1] * V_vec[0])
        los_rate = cross / (dist**2 + eps)

        LAMBDA_DOT_MAX = 10.0
        los_norm = abs(los_rate) / LAMBDA_DOT_MAX
        los_norm = float(np.clip(los_norm, 0.0, 1.0))

        r_los = self.W_LOS * (1.0 - los_norm)

        reward = r_close + r_los - self.STEP_PENALTY
     
        return float(reward), False

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

    def _get_obs(self, agent):
        # 1. 내 상태 (위치, 속도)
        my_state = self.client.getMultirotorState(vehicle_name=agent)
        
        # World Frame 속도 (Yaw=0 고정이므로 Body Frame과 동일 취급)
        v_world_x = my_state.kinematics_estimated.linear_velocity.x_val
        v_world_y = my_state.kinematics_estimated.linear_velocity.y_val

        norm_vx = np.clip(v_world_x / self.MAX_SPEED, -1.0, 1.0)
        norm_vy = np.clip(v_world_y / self.MAX_SPEED, -1.0, 1.0)

        # 2. 적군(DynamicObstacle) 상태
        # PN 보상 계산용 변수들 (Closing speed, LOS rate)
        # -----------------------------------------------------------
        my_pos = np.array([
            my_state.kinematics_estimated.position.x_val,
            my_state.kinematics_estimated.position.y_val
        ], dtype=np.float32)
        my_vel_global = np.array([v_world_x, v_world_y], dtype=np.float32)

        target_state = self.client.getMultirotorState(vehicle_name=self.dynamic_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val
        ], dtype=np.float32)
        target_vel = np.array([
            target_state.kinematics_estimated.linear_velocity.x_val,
            target_state.kinematics_estimated.linear_velocity.y_val
        ], dtype=np.float32)

        R_vec = target_pos - my_pos
        V_vec = target_vel - my_vel_global
        dist = float(np.linalg.norm(R_vec))
        epsilon = 1e-6

        closing_speed = -float(np.dot(R_vec, V_vec)) / (dist + epsilon)
        cross_prod = float(R_vec[0] * V_vec[1] - R_vec[1] * V_vec[0])
        los_rate = cross_prod / (dist**2 + epsilon)

        closing_norm = float(np.clip(closing_speed / 30.0, -1.0, 1.0))
        los_norm = float(np.clip(los_rate / 10.0, -1.0, 1.0))
        # -----------------------------------------------------------

        # [수정] 상대 위치(dx, dy) 관측값 생성
        _leader_feats = []
        _ally_feats = []
        
        # 리더 상대 위치 (dx, dy)
        lx, ly = self._get_relative_pos(agent, "Drone1")
        _leader_feats = [lx, ly]

        # 아군 상대 위치 (dx, dy)
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            ox, oy = self._get_relative_pos(agent, other)
            _ally_feats.append(ox)
            _ally_feats.append(oy)

        # 적군 상대 위치 (dx, dy) 및 PN 정보
        ex, ey = self._get_relative_pos(agent, self.dynamic_name)
        _dynamic_feats = [
            ex, ey,         # 상대 위치 (Vector)
            closing_norm,   # 접근 속도
            los_norm        # 시선각 변화율
        ]

        # 최종 관측값 결합
        obs = np.concatenate([
            np.array(_leader_feats, dtype=np.float32).flatten(),
            np.array(_ally_feats, dtype=np.float32).flatten(),
            np.array(_dynamic_feats, dtype=np.float32).flatten(),
            np.array([norm_vx, norm_vy], dtype=np.float32) # 내 속도
        ]).astype(np.float32)

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
        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        self._reset_obstacle_logic()
        self._get_current_location()

        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._last_pose.clear()
        self._last_time.clear()

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}

        obs_list = [self._get_obs(a) for a in self.agents]
        print("reset.")
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
        self._get_current_location()

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

            per_agent_obs.append(self._get_obs(agent))

            _reward, _ = self._compute_reward(agent)
            per_agent_results.append(_reward)
            per_agent_infos.append({"step_reward": _reward})

        termination_list = [False for _ in self.possible_agents]
        rewards_list = self._get_rewards(per_agent_results)
        obs_list = per_agent_obs
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list