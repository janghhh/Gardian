import gym
import numpy as np
import airsim
import math
import time
import random
from collections import deque

# DBSCAN 및 클러스터링 관련 라이브러리 제거 (사용 안 함)
# from sklearn.cluster import DBSCAN 
# from sklearn.neighbors import NearestNeighbors 

class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",                 # 기본값
        follower_names=("Follower0", "Follower1", "Follower2"),
        lidar_name="LidarSensor",               
        min_samples=5, # <--- 미사용
        step_length=0.01,
        fixed_z=-10.0,
        leader_velocity=1.0,                    
        optimal_distance=10.0,                  
        far_cutoff=60.0,                        
        too_close=0.5,                          
        dt=3.0,                                
        do_visualize=True                       
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]

        # 충돌 관련 설정
        self.COLLISION_THRESHOLD = 1.0 # 모든 거리 기반 충돌 판단 임계값 (m)
        self.STOP_DISTANCE_LEADER_OBSTACLE = 2.0 # 유인기-장애물 충돌 임계값 (m)
        
        # 속도/액션 버퍼
        self.vmax_self = 3.0            
        self._timestep = 1.0

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}
        self._last_time = {}

        # 액션 버퍼
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        self.use_teleport = False  

        # 클라이언트
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = float(fixed_z)
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)
        self.max_cmd_speed = self.step_length / self.dt
        self.leader_velocity = float(leader_velocity)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)

        self._first_setup = True
        self.leader_stop = False

        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1 # 나를 제외한 아군 수
        self.K_enemy = 1                     # 동적 장애물 1대
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy
        
        obs_dim = 3 + 2 + 2 * self.num_ally + 2 * self.num_enemy
        share_obs_dim = obs_dim * len(self.possible_agents)

        low_self = [-20.0, -20.0, -20.0]
        high_self = [20.0, 20.0, 20.0]
        low_bearing = -1.0
        high_bearing = 1.0
        low_dist = 0.0
        high_dist = 200.0

        per_agent_low = low_self + [low_bearing, low_dist] + [low_bearing, low_dist] * self.num_ally + [low_bearing, low_dist] * self.num_enemy
        per_agent_high = high_self + [high_bearing, high_dist] + [high_bearing, high_dist] * self.num_ally + [high_bearing, high_dist] * self.num_enemy

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low, dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,), dtype=np.float32
            ) for agent in self.possible_agents
        }

        self.num_yaw_bins = 9
        self.forward_speed = 3
        self.action_spaces = {
            agent: gym.spaces.Discrete(self.num_yaw_bins)
            for agent in self.possible_agents
        }

        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,), dtype=np.float32
        )

        self.current_waypoint_idx = 0
        self.dynamic_name = "DynamicObstacle"
        self._setup_flight()
        self._generate_leader_waypoints()
        self._last_visualize_t = time.time()

    # ======================================================================
    # 헬퍼 메서드: 포즈/속도/관측 관련
    # ======================================================================
    def _quat_to_rot(self, q):
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        R = np.array([
            [ww + xx - yy - zz, 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), ww - xx + yy - zz, 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), ww - xx - yy + zz]
        ], dtype=np.float32)
        return R

    def _get_pose_xyz(self, name):
        # simGetObjectPose는 'object_name'을 인수로 사용합니다.
        pose = self.client.simGetObjectPose(object_name=name).position
        return np.array([pose.x_val, pose.y_val, pose.z_val], dtype=np.float32)

    def _get_yaw(self, name: str) -> float:
        pose = self.client.simGetVehiclePose(vehicle_name=name)
        _, _, yaw = airsim.to_eularian_angles(pose.orientation)
        return float(yaw)

    def _bearing_and_distance(self, src_pos, src_yaw, tgt_pos):
        dx = float(tgt_pos[0] - src_pos[0])
        dy = float(tgt_pos[1] - src_pos[1])
        dist = math.hypot(dx, dy)
        angle_world = math.atan2(dy, dx)
        bearing = angle_world - src_yaw
        while bearing > math.pi:
            bearing -= 2.0 * math.pi
        while bearing < -math.pi:
            bearing += 2.0 * math.pi
        bearing_norm = bearing / math.pi  # [-1, 1]
        return bearing_norm, dist
    
    def _targets_to_polar_feats(self, agent: str, targets: list, K: int, yaw: float, pos: np.ndarray):
        feats = []
        for i in range(K):
            if i < len(targets):
                target = targets[i]
                if isinstance(target, tuple) and len(target) == 2:
                    cpos = target[1]
                else:
                    cpos = target

                bearing, dist = self._bearing_and_distance(pos, yaw, cpos)
                feats.extend([bearing, dist])
            else:
                feats.extend([0.0, 0.0])  # 패딩
        return np.array(feats, dtype=np.float32)

    def _get_self_velocity(self, name, now_t, current_pos):
        pos = current_pos
        v = np.zeros(3, np.float32)
        if name in self._last_pose:
            dt = max(1e-6, now_t - self._last_time.get(name, now_t))
            v = (pos - self._last_pose[name]) / dt
        self._last_pose[name] = pos
        self._last_time[name] = now_t

        v = np.clip(v, -self.vmax_self, self.vmax_self)
        prev = getattr(self, "_vel_ema_" + name, v)
        v_ema = 0.7 * prev + 0.3 * v
        setattr(self, "_vel_ema_" + name, v_ema)
        return v_ema
    
    # ======================================================================
    # 초기화/이동/시각화 관련
    # ======================================================================
    def _setup_flight(self):
        if self._first_setup:
            self.client.reset()
            self.client.enableApiControl(True, vehicle_name="Drone1")
            self.client.armDisarm(True, vehicle_name="Drone1")
            for agent in self.possible_agents:
                self.client.enableApiControl(True, vehicle_name=agent)
                self.client.armDisarm(True, vehicle_name=agent)

            self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
            self.client.armDisarm(True, vehicle_name=self.dynamic_name)

            futs = [self.client.takeoffAsync(vehicle_name="Drone1")]
            futs += [self.client.takeoffAsync(vehicle_name=a) for a in self.possible_agents]
            futs += [self.client.takeoffAsync(vehicle_name=self.dynamic_name)]
            for f in futs:
                f.join()

            time.sleep(1.0)
            self._teleport_to_start()
            self._first_setup = False

    def _teleport_to_start(self):
        leader_start_x, leader_start_y = 5.0, 2.5
        radius = random.uniform(80.0, 90.0)
        angle = random.uniform(0, 2 * np.pi)
        obstacle_start_x = leader_start_x + radius * np.cos(angle)
        obstacle_start_y = leader_start_y + radius * np.sin(angle)

        start_cfg = {
            "Drone1": (5.0, 2.5, self.fixed_z),
            "Follower0": (0.0, 0.0, self.fixed_z),
            "Follower1": (0.0, 2.5, self.fixed_z),
            "Follower2": (0.0, 5.0, self.fixed_z),
            self.dynamic_name: (obstacle_start_x, obstacle_start_y, self.fixed_z),
        }

        self.client.enableApiControl(True, vehicle_name="Drone1")
        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)

        for name, (x, y, z) in start_cfg.items():
            px, py, pz = float(x), float(y), float(z)
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(px, py, pz),
                    airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)
                ),
                ignore_collision=True,
                vehicle_name=name
            )
        time.sleep(0.05)

    def _generate_leader_waypoints(self):
        leader_start_pos = np.array([5.0, 2.5, self.fixed_z])
        distance = random.uniform(110.0, 130.0)
        angle = random.uniform(0, 2 * np.pi)
        final_destination = leader_start_pos + np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            0.0
        ])
        self.leader_waypoints = [final_destination]
        self.current_waypoint_idx = 0
        
        object_name = "target1v1_5"
        try:
            flag_position = airsim.Vector3r(
                float(final_destination[0]),
                float(final_destination[1]),
                float(self.fixed_z + 8.0)
            )
            flag_orientation = airsim.to_quaternion(0, 80.1, 0)
            flag_pose = airsim.Pose(flag_position, flag_orientation)
            self.client.simSetObjectPose(object_name, flag_pose)
        except Exception as e:
            print(f"'{object_name}' 객체 이동 실패(언리얼에 없을 수 있음): {e}")

    def _update_leader_movement(self):
        """
        유인기를 'Z축 위치 고정' 명령으로 호버링하도록 수정
        """
        
        # 1. Z축 '위치'를 직접 제어하는 moveByVelocityZAsync 사용
        self.client.moveByVelocityZAsync(
            0.0, 0.0,
            float(self.fixed_z), # 0이 아닌, 목표 고도(예: -20.0)를 지정
            duration=self.dt, 
            vehicle_name="Drone1"
        )

        # 2. ★ 수정된 부분: 시각화 주기를 늘리거나 비활성화 (학습 속도 개선 목적)
        if self.do_visualize:
            now = time.time()
            # 0.5초 간격으로 시각화 주기를 늘려 GPU/CPU 부하 감소
            if (now - self._last_visualize_t) >= 0.5: 
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

        # 3. 유인기가 움직이지 않으므로 미션 성공(True)은 반환하지 않음


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
    def _formation_reward(self, agent_pos, leader_pos):
        rel = leader_pos - agent_pos
        dist = math.hypot(float(rel[0]), float(rel[1]))
        if dist < 0.5 or dist > 60.0:
            return -5.0
        ideal = 10.0
        sigma = 10.0
        r = 1.5 * math.exp(-((dist - ideal) ** 2) / (2.0 * sigma ** 2)) - 0.5
        return r

    def _guardian_reward(self, agent_pos, leader_pos, dynamic_pos):
        d_lo = np.linalg.norm(leader_pos[:2] - dynamic_pos[:2])
        d_ao = np.linalg.norm(agent_pos[:2] - dynamic_pos[:2])

        ALERT_DIST = 80.0
        if d_lo > ALERT_DIST:
            return 0.0

        if d_ao < d_lo:
            score = (d_lo - d_ao) / max(d_lo, 1e-3)
            return 5.0 * score
        else:
            return -2.0

    def _compute_reward(self, agent, all_poses, dynamic_pos):
        agent_pos = all_poses[agent]
        leader_pos = all_poses["Drone1"]
        
        # 1) 유인기에 너무 가까움 → 큰 패널티 + 종료 (거리 기반)
        if np.linalg.norm(agent_pos[:2] - leader_pos[:2]) < 1.0:
            return -200.0, True 

        # 3) 포메이션 보상
        r_form = self._formation_reward(agent_pos, leader_pos)

        # 4) 가디언 위치 보상
        r_guard = self._guardian_reward(agent_pos, leader_pos, dynamic_pos)

        # 5) 리더가 이미 장애물에 맞아 멈춘 상태라면 패널티 주고 종료
        if self.leader_stop:
            return -150.0, True

        r_total = r_form + r_guard
        return float(r_total), False

    def _team_reward_and_done(
        self,
        per_agent_results,
        intercepted_agent=None,
        hit_leader=False,
        failed_status=None  # <-- 함수 시그니처에 추가
    ):
        
        if failed_status is not None:
            # 에이전트 충돌로 인한 즉시 종료 (step에서 호출됨)
            return -800.0, True, {"final_status": failed_status}

        if hit_leader:
            # 장애물이 유인기에 닿음 → 최악의 실패 (종료 조건 2)
            return -800.0, True, {"final_status": "FAIL_HIT_LEADER"}

        if intercepted_agent is not None:
            # 장애물이 팔로워 드론에 먼저 부딪힘 (요격) -> 보상만 주고 계속 진행
            tracking_rewards = [r_i for (r_i, _) in per_agent_results]
            base = float(np.mean(tracking_rewards))
            return base + 1000.0, False, { 
                "final_status": "CONTINUE_INTERCEPT",
                "interceptor": intercepted_agent,
            }

        tracking_rewards = [r_i for (r_i, _) in per_agent_results]
        return float(np.mean(tracking_rewards)), False, {}
    
    def _end_episode(self, reward, status):
        """
        에피소드 종료 헬퍼 (충돌 이벤트 발생 시)
        """
        n = len(self.agents)
        rewards_list = [reward] * n
        dones_list = [True] * n
        
        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1")
        all_poses[self.dynamic_name] = self._get_pose_xyz(self.dynamic_name)
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent)

        now_t = self._timestep * self.dt
        obs_list = [self._get_obs(a, now_t, all_poses) for a in self.agents]
        
        infos_list = [{"final_status": status, "reward": reward}] * n
        
        return obs_list, rewards_list, dones_list, infos_list


    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self, t):
        name = self.dynamic_name
        fixed_z = self.fixed_z
        attack_speed = 2.5
        STOP_DISTANCE = self.STOP_DISTANCE_LEADER_OBSTACLE 

        # 상태 초기화
        if not hasattr(self, "_obstacle_state"):
            self._obstacle_state = "IDLE"
            self._next_chase_time = time.time() + random.uniform(0.0, 1.0)
            self._idle_pos = None
            self._chase_mode = None

        # 현재 위치 정보 가져오기 (불필요한 리더 위치 조회 제거 가능하지만, 거리 체크 위해 유지)
        leader_pose = self.client.simGetObjectPose("Drone1").position
        obstacle_pose = self.client.simGetObjectPose(name).position
        lx, ly, lz = leader_pose.x_val, leader_pose.y_val, leader_pose.z_val
        cx, cy, cz = obstacle_pose.x_val, obstacle_pose.y_val, obstacle_pose.z_val

        dx, dy = lx - cx, ly - cy
        dist_2d = math.sqrt(dx ** 2 + dy ** 2) + 1e-9

        # ---------------------------------------------------------
        # 1. IDLE 상태
        # ---------------------------------------------------------
        if self._obstacle_state == "IDLE":
            if self._idle_pos is None:
                radius = random.uniform(90.0, 100.0)
                angle = random.uniform(0, 2 * np.pi)
                self._idle_pos = (lx + radius * np.cos(angle), ly + radius * np.sin(angle))

            ix, iy = self._idle_pos
            dx_i, dy_i = ix - cx, iy - cy
            dist_idle = math.sqrt(dx_i ** 2 + dy_i ** 2)
            
            if dist_idle > 1.0:
                vx = dx_i / dist_idle * 2.0
                vy = dy_i / dist_idle * 2.0
                vz = (fixed_z - cz) * 3.0
                self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration=1.0, vehicle_name=name)
            else:
                self.client.moveByVelocityAsync(0.0, 0.0, 0.0, duration=1.0, vehicle_name=name)

            # 추격 시간 도달 시 -> 상태 변경 및 ★방향 고정★
            if time.time() > self._next_chase_time:
                self._obstacle_state = "CHASE"
                self._chase_start = time.time()
                
                # ★★★ [핵심 수정] CHASE 진입 시점에 방향 벡터를 딱 한 번만 계산하고 저장 ★★★
                # 리더를 향한 단위 벡터 계산
                dir_x = dx / dist_2d
                dir_y = dy / dist_2d
                
                # 속도 벡터 확정 및 저장
                self._chase_vx = dir_x * attack_speed
                self._chase_vy = dir_y * attack_speed
                
                print(f"[추격 시작] 고정 속도 벡터: ({self._chase_vx:.2f}, {self._chase_vy:.2f})")
                return

        # ---------------------------------------------------------
        # 2. CHASE 상태 (계산 없이 저장된 속도로 이동)
        # ---------------------------------------------------------
        elif self._obstacle_state == "CHASE":
            if dist_2d <= STOP_DISTANCE:
                print(f"💥[리더 피격] 충돌 임계점 도달! 에피소드 종료.")
                self.client.moveByVelocityAsync(0.0, 0.0, 0.0, duration=1.0, vehicle_name=name)
                self.leader_stop = True
                return

            # ★★★ 재계산 로직 제거함. 저장된 값(_chase_vx, _chase_vy)을 그대로 사용 ★★★
            # 고도는 높이를 맞춰야 하므로 계속 계산 (단순 연산이라 부하 적음)
            vz = (fixed_z - cz) * 8.0
            
            # 매 스텝 동일한 속도 명령 전송 (duration을 길게 줬으므로 부드럽게 이어짐)
            self.client.moveByVelocityAsync(
                float(self._chase_vx), 
                float(self._chase_vy), 
                float(vz), 
                duration=1.0, 
                vehicle_name=name
            )

        # ---------------------------------------------------------
        # 3. RETURN 상태
        # ---------------------------------------------------------
        elif self._obstacle_state == "RETURN":
            if self._idle_pos is None:
                radius = random.uniform(90.0, 100.0)
                angle = random.uniform(0, 2 * np.pi)
                self._idle_pos = (lx + radius * math.cos(angle), ly + radius * math.sin(angle))

            ix, iy = self._idle_pos
            dx_r, dy_r = ix - cx, iy - cy
            dist_return = math.sqrt(dx_r ** 2 + dy_r ** 2)
            
            if dist_return > 1.0:
                vx = dx_r / dist_return * 4.0
                vy = dy_r / dist_return * 4.0
                vz = (fixed_z - cz) * 8.0
                self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration=1.0, vehicle_name=name)
            else:
                self._obstacle_state = "IDLE"
                self._next_chase_time = time.time() + random.uniform(1.0, 3.0)

    def _teleport_obstacle_around_leader(self):

        # 1. 리더의 현재 위치를 조회하지 않고, 에피소드 시작 고정 좌표 사용
        leader_start_x, leader_start_y = 5.0, 2.5 
        
        # 2. 시작 시점과 동일한 반경 및 각도 설정
        radius = random.uniform(80.0, 90.0)
        angle  = random.uniform(0, 2 * np.pi)
        
        # 3. 좌표 계산
        ox = leader_start_x + radius * math.cos(angle)
        oy = leader_start_y + radius * math.sin(angle)
        oz = self.fixed_z

        # 4. 위치 설정 (float 형변환 유지)
        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(float(ox), float(oy), float(oz)), 
                airsim.Quaternionr(0, 0, 0, 1)
            ),
            ignore_collision=True,
            vehicle_name=self.dynamic_name
        )
        
        # 5. 순간이동 후 속도 0으로 초기화 및 정지 (스텝 시간 dt 동안 유지)
        self.client.moveByVelocityAsync(0.0, 0.0, 0.0, duration=self.dt, vehicle_name=self.dynamic_name)
        
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

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._setup_flight()
        self._generate_leader_waypoints()
        self.current_waypoint_idx = 0
        self._teleport_to_start()
        self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
        self.client.simFlushPersistentMarkers()

        self._timestep = 0
        self._last_pose.clear()
        self._last_time.clear()
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}
        self.leader_stop = False # 피격 플래그 초기화

        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1")
        all_poses[self.dynamic_name] = self._get_pose_xyz(self.dynamic_name)
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent)

        now_t = 0.0
        obs_list = [self._get_obs(a, now_t, all_poses) for a in self.agents]

        self._obstacle_state = "IDLE"
        self._idle_pos = None
        self._next_chase_time = time.time() + random.uniform(1.0, 3.0)
        self._chase_mode = None
        
        return obs_list

    def _get_obs(self, agent, now_t, all_poses):
        now_t = self._timestep * self.dt

        agent_pos = all_poses[agent]
        leader_pos = all_poses["Drone1"]
        dynamic_pos = all_poses[self.dynamic_name] # 동적 장애물 (적군) 위치

        self_vel = self._get_self_velocity(agent, now_t, agent_pos)
        yaw = self._get_yaw(agent)

        bearing_leader, dist_leader = self._bearing_and_distance(agent_pos, yaw, leader_pos)

        # 1. 아군 피처 (나를 제외한 다른 팔로워들)
        ally_positions = [all_poses[other_agent] for other_agent in self.agents if other_agent != agent]
        ally_feats = self._targets_to_polar_feats(agent, ally_positions, self.num_ally, yaw, agent_pos)

        # 2. 적군 피처 (동적 장애물)
        enemy_positions = [dynamic_pos]
        enemy_feats = self._targets_to_polar_feats(agent, enemy_positions, self.num_enemy, yaw, agent_pos)

        obs = np.concatenate(
            [
                self_vel.astype(np.float32),
                np.array([bearing_leader, dist_leader], dtype=np.float32),
                ally_feats,
                enemy_feats,
            ],
            axis=0
        ).astype(np.float32)

        return obs

    def _do_action(self, agent, action):
        if isinstance(action, (np.ndarray, list, tuple)):
            a_idx = int(action[0])
        else:
            a_idx = int(action)
        a_idx = np.clip(a_idx, 0, self.num_yaw_bins - 1)

        ratio = a_idx / (self.num_yaw_bins - 1) if self.num_yaw_bins > 1 else 0.5
        yaw_norm = -1.0 + 2.0 * ratio  # [-1, 1]

        self._last_action[agent] = np.array([yaw_norm, 0.0], dtype=np.float32)
        self._current_action[agent] = self._last_action[agent].copy()

        max_yaw_rate_deg = 90.0  # deg/s
        yaw_rate = math.radians(max_yaw_rate_deg) * yaw_norm  # [rad/s]

        # ----------------------------------------------------------------------
        # ★ 수정된 부분: 단일 API 호출로 통합 (이전의 moveByRollPitchYawrateZAsync.join() 제거)
        # ----------------------------------------------------------------------
        
        # 현재 Yaw 각도 읽기
        pose = self.client.simGetVehiclePose(vehicle_name=agent)
        _, _, current_yaw = airsim.to_eularian_angles(pose.orientation)

        speed = self.forward_speed  # [m/s]
        
        # 목표 Yaw는 현재 Yaw에서 회전율을 적용하여 dt 시간 후의 각도를 추정
        # moveByVelocityZAsync에 Yaw Rate 모드를 사용하여 회전과 전진을 동시에 지시
        
        # 현재 Yaw 각도와 속도를 이용해 VX, VY 계산 (Forward Speed)
        vx = speed * math.cos(current_yaw)
        vy = speed * math.sin(current_yaw)

        # moveByVelocityZAsync를 사용하여 속도와 Yaw Rate를 동시에 지정
        # **duration=self.dt**로 설정하여 전체 스텝 시간 동안 명령을 유지
        self.client.moveByVelocityZAsync(
            vx=vx, vy=vy, z=float(self.fixed_z),
            duration=self.dt, 
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            # is_rate=True로 Yaw Rate를 적용하여 회전을 지시
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate)),
            vehicle_name=agent
        )
        # 이 비동기 호출 후 .join()을 호출하지 않으므로, 병렬 에이전트 처리가 빨라집니다.
    
    def step(self, actions):
        self._timestep += 1
        t = time.time()
        
        # 1) 에이전트 액션 적용
        for agent, act in zip(self.agents, actions):
            self._do_action(agent, act)

        # 2) 유인기/장애물 이동
        self._update_leader_movement()
        self._update_dynamic_obstacle(t) 

        # 3) 위치들 수집 (이동 후 위치)
        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1")
        all_poses[self.dynamic_name] = self._get_pose_xyz(self.dynamic_name)
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent)
        dynamic_pos = all_poses[self.dynamic_name]
        leader_pos = all_poses["Drone1"]
        

        # 🔥 (0) 이탈 체크: far_cutoff (초록선) 초과 시 즉시 종료
        for agent in self.agents:
            agent_pos = all_poses[agent]
            # 2D 거리 계산
            dist_to_leader = np.linalg.norm(agent_pos[:2] - leader_pos[:2])
            
            if dist_to_leader > self.far_cutoff:
                print(f"❌[이탈] {agent}이 리더와의 거리({dist_to_leader:.2f}m)가 이탈 임계값({self.far_cutoff}m) 초과! → 전체 실패")
                print(f"[현재 스텝: {self._timestep}]")
                # -1000.0의 큰 패널티를 부여하고 에피소드 종료
                return self._end_episode(-1000.0, "FAIL_AGENT_FAR_CUTOFF")

        # ================================
        # 🔥 (0) 거리 기반 충돌/종료 체크 (우선순위 높음)
        # ================================
        intercepted_agent = None
        
        for i in range(len(self.agents)):
            agent_i = self.agents[i]
            pos_i = all_poses[agent_i]
            
            # A) Agent-Agent Collision (거리 기반 - 종료)
            for j in range(i + 1, len(self.agents)):
                agent_j = self.agents[j]
                pos_j = all_poses[agent_j]
                dist_aa = np.linalg.norm(pos_i[:2] - pos_j[:2])
                
                if dist_aa < self.COLLISION_THRESHOLD:
                    print(f"💥[거리 충돌] {agent_i} ↔ {agent_j} ({dist_aa:.2f}m) → 전체 실패")
                    print(f"[현재 스텝: {self._timestep}]")
                    return self._end_episode(-1000.0, "FAIL_AGENT_COLLISION")

            # B) Agent-Leader Collision (거리 기반 - 종료)
            dist_al = np.linalg.norm(pos_i[:2] - leader_pos[:2])
            if dist_al < self.COLLISION_THRESHOLD:
                print(f"💥[거리 충돌] {agent_i}이 Drone1에 부딪힘 ({dist_al:.2f}m) → 전체 실패")
                print(f"[현재 스텝: {self._timestep}]")
                return self._end_episode(-1000.0, "FAIL_AGENT_HIT_LEADER")
                
            # C) Agent-Obstacle Collision (거리 기반 요격 - 계속 진행)
            dist_ao = np.linalg.norm(pos_i[:2] - dynamic_pos[:2])
            if dist_ao < self.COLLISION_THRESHOLD and intercepted_agent is None:
                # ** 요격 성공 (거리): 보상 후 장애물 순간이동 **
                print(f"🛡️[거리 가디언] {agent_i}이 동적장애물({self.dynamic_name})을 요격! ({dist_ao:.2f}m)")
                print(f"[현재 스텝: {self._timestep}]")
                self._teleport_obstacle_around_leader()
                intercepted_agent = agent_i
                
                # 장애물 위치 즉시 갱신
                all_poses[self.dynamic_name] = self._get_pose_xyz(self.dynamic_name)
                dynamic_pos = all_poses[self.dynamic_name]
                
        # ================================
        # 🔥 (1) AirSim 충돌 이벤트 검사 (물리적 접촉 확인)
        # ================================
        for agent in self.agents:
            col = self.client.simGetCollisionInfo(vehicle_name=agent)
            if col.has_collided and col.object_name == self.dynamic_name:
                # Agent ↔ Obstacle 물리 충돌 (요격)
                if intercepted_agent is None: # 거리 체크에서 이미 처리되지 않았을 경우에만 처리
                    print(f"🛡️[이벤트 가디언] {agent}이 동적장애물({self.dynamic_name})을 막음!")
                    print(f"[현재 스텝: {self._timestep}]")
                    self._teleport_obstacle_around_leader()
                    intercepted_agent = agent
        
        # ================================
        # 🔥 (2) Agent - Leader 물리 충돌 이벤트 검사 (새로 추가)
        # ================================
        for agent in self.agents:
            col = self.client.simGetCollisionInfo(vehicle_name=agent)
            
            # 충돌이 발생했고, 그 대상이 유인기("Drone1")인 경우
            if col.has_collided and col.object_name == "Drone1":
                print(f"💥[이벤트 충돌] {agent}이 유인기(Drone1)와 물리 충돌 발생!")
                print(f"[현재 스텝: {self._timestep}]")
                # 물리 충돌 발생 시 거리 충돌과 동일한 큰 패널티로 즉시 에피소드 종료
                return self._end_episode(-1000.0, "FAIL_AGENT_PHYSICAL_HIT_LEADER")
                    
        # 4) 리더 피격 여부 체크 (_update_dynamic_obstacle에서 설정됨)
        hit_leader = self.leader_stop

        # 5) 관측 / 개별 보상 
        obs_list, per_agent_results, per_agent_infos = [], [], []
        now_t = self._timestep * self.dt
        for agent in self.agents:
            o = self._get_obs(agent, now_t, all_poses)
            r, done_i = self._compute_reward(agent, all_poses, dynamic_pos) 

            obs_list.append(o)
            per_agent_results.append((float(r), bool(done_i)))
            per_agent_infos.append({"reward": float(r)})

        # 6) 팀 보상 / 종료 판정
        team_reward, done_all, final_team_info = self._team_reward_and_done(
            per_agent_results,
            intercepted_agent=intercepted_agent, 
            hit_leader=hit_leader,               
            failed_status=None, # 즉시 종료는 이미 위에서 처리했으므로 None
        )

        n = len(self.agents)
        rewards_list = [team_reward] * n
        dones_list = [done_all] * n
        infos_list = []
        for i in range(n):
            info_i = per_agent_infos[i].copy()
            info_i.update(final_team_info)
            infos_list.append(info_i)

        return obs_list, rewards_list, dones_list, infos_list
    
    