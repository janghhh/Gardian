# from MARL_test import ParallelEnv
import gym
import numpy as np
import airsim
import math
import time
import random
from sklearn.cluster import DBSCAN


class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",                 # 기본값
        follower_names=("Follower0", "Follower1", "Follower2"),
        lidar_name="LidarSensor",               # 라이다 공통명 (실제 센서는 f"{agent}_{lidar_name}")
        min_samples=5,
        step_length=1.0,
        fixed_z=-20.0,
        leader_velocity=1.0,                    # 유인기 속도(m/s) 파라미터화
        optimal_distance=10.0,                  # 추종 최적 거리(시각화용 링)
        far_cutoff=60.0,                        # 이탈 종료 거리(시각화용 링)
        too_close=0.5,                          # 유인기와 최소 거리
        dt=0.1,                                # 내부 시뮬레이션 타임스텝(초)
        do_visualize=True                       # 원/포인트 시각화 on/off
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]

        # 라이다 / DBSCAN
        self.lidar_name = lidar_name
        self.min_samples = int(min_samples)
        self.eps = 0.3
        self.min_pts = 5

        # 관측공간(동적 객체 K-NN)
        self.max_enemies = 1    
        self.max_allies = len(self.possible_agents) - 1            
        self.match_dist_max = 2.0           # 프레임 간 클러스터 매칭 허용 거리[m]
        self._cluster_tracks = {a: {} for a in self.possible_agents}  # agent별 트랙 사전
        self._next_cluster_id = {a: 0 for a in self.possible_agents}  # agent별 ID 증가기
        self.enemy_clusters = {a: [] for a in self.possible_agents}   # 적군 클러스터 리스트
        self.ally_clusters = {a: [] for a in self.possible_agents}    # 아군 클러스터 리스트

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}     # {"name": np.array([x,y,z])}
        self._last_time = {}     # {"name": t_float}

        # 속도/액션 버퍼
        self.vmax_self = 10.0             # 자기 속도 상한(m/s) (학습 안정용)
        self.vmax_cluster = 15.0
        self.enemy_speed_threshold = 0.5  #  클러스터 속도 임계값 (m/s)
        self._timestep = 0

        # KeyError 방지: 액션 버퍼를 에이전트별로 초기화
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        # 이동 모드: 이제 텔레포트 대신 moveByVelocity 기반
        self.use_teleport = False  # False면 위치 차분으로 속도 추정

        # 클라이언트
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = float(fixed_z)
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)

        # 액션 → 속도[m/s] 변환 스케일
        # 결과: 한 스텝 이동 거리 = 10.0 * 0.1 = 1.0m
        self.agent_constant_speed = 10.0

        self.leader_velocity = float(leader_velocity)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)

        self._first_setup = True
        self.leader_stop = False

        # ===== obs / act / share_obs spaces (NEW: Bearing/Distance 기반) =====
        act_dim = 2

        # 1. 유인기 (H_Bearing, V_Bearing, 3D_Dist): 3
        low_leader = [-1.0, -1.0, 0.0]
        high_leader = [1.0, 1.0, 200.0]
        
        # 2. 모든 아군 (H_Bearing, V_Bearing, 3D_Dist): max_allies * 3 (2 * 3 = 6)
        low_allies = [-1.0, -1.0, 0.0] * self.max_allies
        high_allies = [1.0, 1.0, 200.0] * self.max_allies
        
        # 3. 라이더 범위 내 적군 (H_Bearing, V_Bearing, 3D_Dist): max_enemies * 3 (1 * 3 = 3)
        low_enemies = [-1.0, -1.0, 0.0] * self.max_enemies
        high_enemies = [1.0, 1.0, 200.0] * self.max_enemies

        # 4. 경계 포인트 (H_Bearing, V_Bearing=0, 2D_Dist): 3
        # (가장 가까운 LiDAR 2D 스캔 포인트)
        low_boundary = [-1.0, 0.0, 0.0]
        high_boundary = [1.0, 0.0, self.far_cutoff] # 2D 거리, V_Bearing은 0 고정

        # obs_dim = [Leader(3) + Allies(6) + Enemies(3) + Boundary(3)]
        obs_dim_list = low_leader + low_allies + low_enemies + low_boundary
        obs_dim = len(obs_dim_list) # 3 + 6 + 3 + 3 = 15

        per_agent_low = np.array(obs_dim_list, dtype=np.float32)
        per_agent_high = np.array(high_leader + high_allies + high_enemies + high_boundary, dtype=np.float32)

        assert len(per_agent_low) == obs_dim, f"low len {len(per_agent_low)} != obs_dim {obs_dim}"
        assert len(per_agent_high) == obs_dim, f"high len {len(per_agent_high)} != obs_dim {obs_dim}"

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=per_agent_low,
                high=per_agent_high,
                shape=(obs_dim,), dtype=np.float32
            ) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # (MODIFIED) share_obs_dim 수정
        share_obs_dim = obs_dim * len(self.possible_agents) 
        
        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low.tolist() * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high.tolist() * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,), dtype=np.float32
        )

        self.current_waypoint_idx = 0
        self.dynamic_name = "DynamicObstacle"
        self._setup_flight()
        # self._generate_leader_waypoints()
        self._last_visualize_t = time.time()

        # 디버깅 플래그
        self.debug_clusters = True

    # ======================================================================
    # 라이다 데이터 → DBSCAN → 트랙/속도 추정
    # ======================================================================
    def _lidar_sensor_name(self, agent: str) -> str:
        # 실제 AirSim 설정: 각 드론에 "Follower0_LidarSensor" 식으로 붙어 있다고 가정
        return f"{agent}_{self.lidar_name}"

    def _LidarPointsWorld(self, agent, data_frame="VehicleInertialFrame"):
        ld = self.client.getLidarData(lidar_name=self._lidar_sensor_name(agent), vehicle_name=agent)
        arr = np.array(ld.point_cloud, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        pts = arr.reshape(-1, 3)
        return pts

    def _ClusterDbscanWorld(self, agent):
        pts_w = self._LidarPointsWorld(agent, data_frame="VehicleInertialFrame")
        if pts_w.shape[0] == 0:
            return []
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pts_w)
        labels = db.labels_
        clusters = []
        for cid in np.unique(labels):
            if cid == -1:
                continue
            idx = (labels == cid)
            cpts = pts_w[idx]
            centroid = cpts.mean(axis=0)
            clusters.append({"centroid": centroid, "count": int(idx.sum())})
        return clusters

    def _update_clusters_and_tracks(self, agent, now_t):
        tracks = self._cluster_tracks[agent]
        clusters = self._ClusterDbscanWorld(agent)

        cur_centroids = [c["centroid"] for c in clusters]
        cur_used = [False] * len(cur_centroids)

        # ===== 1) 기존 트랙 업데이트 / 삭제 =====
        for tid, st in list(tracks.items()):
            best_j, best_d = -1, 1e9
            for j, cen in enumerate(cur_centroids):
                if cur_used[j]:
                    continue
                d = np.linalg.norm(cen - st["centroid"])
                if d < best_d:
                    best_d, best_j = d, j

            if best_j >= 0 and best_d <= self.match_dist_max:
                dt = max(1e-6, now_t - st["last_t"])
                new_c = cur_centroids[best_j]
                vel = (new_c - st["centroid"]) / dt
                # 클램프 + EMA
                vel = np.clip(vel, -self.vmax_cluster, self.vmax_cluster)
                prev = st.get("vel", np.zeros(3, np.float32))
                #vel = 0.6 * prev + 0.4 * vel
                alpha = 0.1 # 또는 0.2
                vel = (1.0 - alpha) * prev + alpha * vel
                tracks[tid] = {"centroid": new_c, "vel": vel, "last_t": now_t}
                cur_used[best_j] = True
            else:
                if now_t - st["last_t"] > 1.0:
                    del tracks[tid]

        # ===== 2) 매칭 안 된 클러스터는 신규 트랙 생성 =====
        for j, cen in enumerate(cur_centroids):
            if not cur_used[j]:
                tid = self._next_cluster_id[agent]
                self._next_cluster_id[agent] += 1
                tracks[tid] = {"centroid": cen,
                               "vel": np.zeros(3, np.float32),
                               "last_t": now_t}

        # ===== 3) 적군 / 아군 클러스터 분류 (★ 상대속도 기준) =====
        agent_pos = self._get_pose_xyz(agent)
        leader_pos = self._get_pose_xyz("Drone1")

        # 에이전트 / 리더 속도 추정
        v_agent = self._get_self_velocity(agent, now_t, agent_pos)
        v_leader = self._get_self_velocity("Drone1", now_t, leader_pos)

        # “기준 속도”: 리더-에이전트 상대속도 (리더보다 더 요란하게 움직이면 적군으로)
        rel_speed_leader_agent = np.linalg.norm(v_leader - v_agent)

        enemy_list = []
        ally_list = []

        for tid, st in tracks.items():
            v_cluster = st["vel"]

            # 에이전트 기준 클러스터 상대속도
            rel_speed_cluster_agent = float(np.linalg.norm(v_cluster - v_agent))

            entry = {
                "id":       tid,
                "centroid": st["centroid"].copy(),
                "vel":      v_cluster.copy(),
                "last_t":   st["last_t"],
                # 디버그 출력용: 이제 진짜 '에이전트 기준 상대속도'
                "rel_speed": rel_speed_cluster_agent,
            }

            # 방법 1) 리더보다 더 요란하게 움직이면 적군
            if rel_speed_cluster_agent > rel_speed_leader_agent :
                enemy_list.append(entry)
            else:
                ally_list.append(entry)

            # 방법 2) 단순 임계값으로 하고 싶으면 아래처럼:
            #if rel_speed_cluster_agent > self.enemy_speed_threshold:
            #     enemy_list.append(entry)
            #else:
            #     ally_list.append(entry)
        # DynamicObstacle 실제 위치
        # dyn_pos = self._get_pose_xyz(self.dynamic_name)

        # 각 클러스터가 동적장애물에서 얼마나 떨어져 있는지 출력
        # for e in enemy_list[:3] + ally_list[:3]:
        #     d_dyn = np.linalg.norm(e["centroid"] - dyn_pos)
        #     print(f"  -> cluster {e['id']} dist_to_dynamic={d_dyn:.2f}")


        def _dist(e):
            return np.linalg.norm(e["centroid"] - agent_pos)

        enemy_list.sort(key=_dist)
        ally_list.sort(key=_dist)

        self.enemy_clusters[agent] = enemy_list
        self.ally_clusters[agent] = ally_list


    # ======================================================================
    # 디버깅용: 콘솔에 적/아군 클러스터 출력
    # ======================================================================
    def _debug_print_clusters(self):
        if not self.debug_clusters:
            return

        # 너무 많이 찍히지 않게, 10스텝마다 한 번만 출력
        if self._timestep % 3 != 0:
            return

        print(f"\n[step {self._timestep}] ==== 클러스터 디버그 ====")
        for agent in self.agents:
            enemies = self.enemy_clusters.get(agent, [])
            allies = self.ally_clusters.get(agent, [])

            print(f"▶ {agent}: 적군 {len(enemies)}개, 아군 {len(allies)}개")

            # 적군 최대 2개까지만 자세히 출력
            for e in enemies[:5]:
                c = e["centroid"]
                v = e["vel"]
                print(f"[ENEMY] id={e['id']}, pos=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}), "
                      f"vel=({v[0]:.1f},{v[1]:.1f},{v[2]:.1f}), rel_speed={e['rel_speed']:.2f}")

            # 아군도 최대 2개까지만
            for a in allies[:5]:
                c = a["centroid"]
                v = a["vel"]
                print(f"[ALLY ] id={a['id']}, pos=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}), "
                      f"vel=({v[0]:.1f},{v[1]:.1f},{v[2]:.1f}), rel_speed={a['rel_speed']:.2f}")

    # ======================================================================
    # 포즈/속도
    # ======================================================================
    def _get_pose_xyz(self, name):
        pose = self.client.simGetVehiclePose(vehicle_name=name).position
        return np.array([pose.x_val, pose.y_val, pose.z_val], dtype=np.float32)

    def _get_self_velocity(self, name, now_t, current_pos):
        """
        [최적화됨] _get_pose_xyz() API 호출을 제거하고,
        미리 계산된 current_pos를 인자로 받습니다.
        """
        if self.use_teleport and name in self._last_action:
            # 텔레포트 모드 (현재는 사용 안 함)
            a = self._last_action[name]
            vx = float(a[0]) * float(self.step_length) / self.dt
            vy = float(a[1]) * float(self.step_length) / self.dt
            vz = 0.0
            v = np.array([vx, vy, vz], dtype=np.float32)
        else:
            # moveByVelocity 기반: 위치 차분으로 속도 추정
            pos = current_pos
            v = np.zeros(3, np.float32)
            if name in self._last_pose:
                dt = max(1e-6, now_t - self._last_time.get(name, now_t))
                v = (pos - self._last_pose[name]) / dt
            self._last_pose[name] = pos
            self._last_time[name] = now_t

        # 학습 안정화를 위한 클램프 & EMA
        v = np.clip(v, -self.vmax_self, self.vmax_self)
        prev = getattr(self, "_vel_ema_" + name, v)
        v_ema = 0.7 * prev + 0.3 * v
        setattr(self, "_vel_ema_" + name, v_ema)
        return v_ema

    def _get_knn_features(self, agent, K, now_t, all_poses):
        """
        all_poses 딕셔너리를 사용해 K-최근접 클러스터의
        상대 위치/상대 속도를 뽑아 관측에 넣는다.
        """
        pos_a = all_poses[agent]
        vel_a = self._get_self_velocity(agent, now_t, pos_a)

        feats = []
        cand = []
        for _, tr in self._cluster_tracks[agent].items():
            rel_pos = tr["centroid"] - pos_a
            rel_vel = tr["vel"] - vel_a
            dist = np.linalg.norm(rel_pos)
            cand.append((dist, rel_pos, rel_vel))
        cand.sort(key=lambda x: x[0])

        for i in range(K):
            if i < len(cand):
                _, rp, rv = cand[i]
                feats.extend([rp[0], rp[1], rp[2], rv[0], rv[1], rv[2]])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 패딩
        return np.array(feats, dtype=np.float32)

    # (NEW) 1D LiDAR 스캔 (경계점 관측용)
    def _get_lidar_obs(self, agent):
        lidar_data = self.client.getLidarData(
            vehicle_name=agent,
            lidar_name=self._lidar_sensor_name(agent)
        )
        if len(lidar_data.point_cloud) < 3:
            return np.full(36, self.far_cutoff, dtype=np.float32)

        pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        # (중요) 2D 거리만 사용
        dists = np.linalg.norm(pts[:, :2], axis=1) 
        angles = np.arctan2(pts[:, 1], pts[:, 0])

        bins = np.linspace(-math.pi, math.pi, 37)
        min_dists = np.full(36, self.far_cutoff, dtype=np.float32)

        for i in range(36):
            sel = dists[(angles >= bins[i]) & (angles < bins[i+1])]
            if len(sel) > 0:
                min_dists[i] = np.min(sel)
        return min_dists


    def _calculate_relative_bearing(self, agent_pos, agent_orientation_quat, target_pos):
        """
        에이전트 기준 타겟의 상대 3D 방위와 3D 거리를 반환
        
        반환: (h_bearing, v_bearing, distance_3d)
            h_bearing: 수평 방위 (-1 ~ 1) (Yaw)
            v_bearing: 수직 방위 (-1 ~ 1) (Pitch)
            distance_3d: 3D 거리 (m)
        """

        # 1. 에이전트 -> 타겟 3D 벡터 (월드 좌표계, NED)
        vec_world = target_pos - agent_pos        # shape: (3,)
        distance_3d = float(np.linalg.norm(vec_world))

        if distance_3d < 1e-6:
            return 0.0, 0.0, 0.0

        # 2. 에이전트 쿼터니언을 회전행렬로 변환 (world ← body)
        #    AirSim Quaternionr: (w, x, y, z)
        w = agent_orientation_quat.w_val
        x = agent_orientation_quat.x_val
        y = agent_orientation_quat.y_val
        z = agent_orientation_quat.z_val

        # R_wb: body → world (NED)
        R_wb = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
            [    2*(x*y + z*w),  1 - 2*(x*x + z*z),      2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y)]
        ], dtype=np.float32)

        # world → body 회전행렬 (역행렬 = 전치행렬)
        R_bw = R_wb.T

        # 3. 월드 좌표계 벡터를 에이전트 로컬(Body) 좌표계로 변환
        vec_local = R_bw @ vec_world   # shape: (3,)

        x_local = float(vec_local[0])   # +X : 앞 (forward, NED)
        y_local = float(vec_local[1])   # +Y : 오른쪽 (right)
        z_ned  = float(vec_local[2])    # +Z : 아래 (down, NED)

        # 우리가 쓰는 방위 표현에서 '위쪽'을 +Z로 쓰고 싶으므로 부호 반전
        z_local = -z_ned                # +Z : 위 (up)

        # 4. 수평 방위 (Yaw) 계산: arctan2(오른쪽, 앞)
        h_angle_rad = np.arctan2(y_local, x_local)   # [-pi, pi]
        h_bearing = h_angle_rad / np.pi              # [-pi, pi] → [-1, 1]

        # 5. 수직 방위 (Pitch) 계산: arctan2(위, 수평거리)
        horizontal_dist = float(np.hypot(x_local, y_local))
        v_angle_rad = np.arctan2(z_local, horizontal_dist)  # [-pi/2, pi/2]
        v_bearing = v_angle_rad / (np.pi / 2.0)             # [-pi/2, pi/2] → [-1, 1]

        return float(h_bearing), float(v_bearing), distance_3d

    # ======================================================================
    # Spaces (PettingZoo-style getters)
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

    # ======================================================================
    # 초기화/이륙/시작 배치
    # ======================================================================
    # (MODIFIED) takeoffAsync 제거
    # (MODIFIED) 3D 이동을 위해 takeoffAsync 모두 제거
    def _setup_flight(self):
        if self._first_setup:
            self.client.reset()
            self.client.enableApiControl(True, vehicle_name="Drone1")
            self.client.armDisarm(True, vehicle_name="Drone1")
            for agent in self.possible_agents:
                self.client.enableApiControl(True, vehicle_name=agent)
                self.client.armDisarm(True, vehicle_name=agent)

            # 동적장애물 제어 추가
            self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
            self.client.armDisarm(True, vehicle_name=self.dynamic_name)

            # (DELETED) futs = [self.client.takeoffAsync(...)] 블록 전체 삭제
            # (DELETED) time.sleep(1.0) 삭제

            self._teleport_to_start() # 텔레포트 함수가 모든 것을 처리
            self._first_setup = False

    # (MODIFIED) 텔레포트 직후 '긴 시간 호버' 명령 추가
    def _teleport_to_start(self):
        
        radius = random.uniform(60.0, 70.0)
        angle = random.uniform(0, 2 * np.pi)
        obstacle_start_x = 0.0 + radius * np.cos(angle)
        obstacle_start_y = 0.0 + radius * np.sin(angle)

        start_cfg = {
            "Drone1": (0, 0, self.fixed_z),
            "Follower0": (10.0, 0.0, self.fixed_z),
            "Follower1": (-5.0, 8.660, self.fixed_z),
            "Follower2": (-5.0, -8.660, self.fixed_z),
            self.dynamic_name: (obstacle_start_x, obstacle_start_y, self.fixed_z),
            }
        
        self.client.enableApiControl(True, vehicle_name="Drone1")
        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
        self.client.enableApiControl(True, vehicle_name=self.dynamic_name) # 누락 방지

        # 1. 텔레포트
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

        # 2. (NEW) '명령 공백'을 메우기 위한 0속도 명령
        # duration을 매우 길게(예: 3600초) 설정하여
        # _do_action에서 첫 명령이 들어올 때까지 호버링 상태를 유지
        duration = 3600.0 
        
        self.client.moveByVelocityAsync(
            0, 0, 0, duration, vehicle_name="Drone1"
        )
        self.client.moveByVelocityAsync(
            0, 0, 0, duration, vehicle_name=self.dynamic_name
        )
        for agent in self.possible_agents:
            self.client.moveByVelocityAsync(
                0, 0, 0, duration, vehicle_name=agent
            )

        time.sleep(0.1) # 0속도 명령이 안정화될 시간

    # ======================================================================
    # 유인기 경로/이동/시각화
    # ======================================================================
    def _generate_leader_waypoints(self):
        leader_start_pos = np.array([5.0, 2.5, self.fixed_z])

        # 시작점으로부터 110~130m 떨어진 곳에 무작위 목적지 설정
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
        Returns:
            mission_accomplished (bool): 항상 False
        """
        
        # 1. (NEW) Z축 '위치'를 직접 제어하는 moveByVelocityZAsync 사용
        #    이 명령이 중력을 가장 확실하게 상쇄합니다.
        self.client.moveByVelocityZAsync(
            0.0, 0.0,
            float(self.fixed_z), # 0이 아닌, 목표 고도(예: -20.0)를 지정
            duration=self.dt, 
            vehicle_name="Drone1"
        )

        # 2. 시각화는 그대로 유지
        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.1:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

        # 3. 유인기가 움직이지 않으므로 미션 성공(True)은 반환하지 않음
        return False

    # --------------------- 시각화 ---------------------
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
    # 관측/액션/보상
    # ======================================================================
    # (MODIFIED) 3D Bearing/Distance 기반 + 클러스터링 활용 (15 Dims)
    def _get_obs(self, agent, now_t, all_poses):
        now_t = self._timestep * self.dt
        
        # 1. 위치 및 3D 방향(쿼터니언) 가져오기
        agent_pos = all_poses[agent]
        leader_pos = all_poses["Drone1"]
        
        # (중요) 에이전트의 현재 3D 방향(쿼터니언)을 API로 직접 가져옴
        agent_pose_full = self.client.simGetVehiclePose(agent)
        agent_quat = agent_pose_full.orientation

        # 2. 유인기 [H_Bearing, V_Bearing, 3D_Dist] (3 dims)
        b_h_l, b_v_l, d_3d_l = self._calculate_relative_bearing(agent_pos, agent_quat, leader_pos)
        leader_feats = np.array([b_h_l, b_v_l, d_3d_l], dtype=np.float32)

        # 3. 아군 [H_Bearing, V_Bearing, 3D_Dist] (max_allies * 3 = 6 dims)
        allies_feats = []
        allies_list = self.ally_clusters.get(agent, []) 
        
        for i in range(self.max_allies):
            if i < len(allies_list):
                target_pos = allies_list[i]["centroid"]
                b_h_a, b_v_a, d_3d_a = self._calculate_relative_bearing(agent_pos, agent_quat, target_pos)
                allies_feats.extend([b_h_a, b_v_a, d_3d_a])
            else:
                allies_feats.extend([0.0, 0.0, 0.0]) # 패딩
        
        # 4. 적군 [H_Bearing, V_Bearing, 3D_Dist] (max_enemies * 3 = 3 dims)
        enemy_feats = []
        enemy_list = self.enemy_clusters.get(agent, [])

        for i in range(self.max_enemies):
            if i < len(enemy_list):
                target_pos = enemy_list[i]["centroid"]
                b_h_e, b_v_e, d_3d_e = self._calculate_relative_bearing(agent_pos, agent_quat, target_pos)
                enemy_feats.extend([b_h_e, b_v_e, d_3d_e])
            else:
                enemy_feats.extend([0.0, 0.0, 0.0]) # 패딩

        # 5. 경계 포인트 [H_Bearing, 0.0, 2D_Dist] (3 dims)
        min_dists_36 = self._get_lidar_obs(agent) # 2D LiDAR 스캔
        
        boundary_dist_2d = np.min(min_dists_36)
        boundary_bin_index = np.argmin(min_dists_36)
        
        boundary_h_bearing = 0.0
        if boundary_dist_2d < self.far_cutoff: # 유효한 포인트가 있을 때만 계산
            bins = np.linspace(-math.pi, math.pi, 37)
            # (중요) LiDAR 각도는 이미 에이전트 기준 상대 각도임
            relative_angle_rad = (bins[boundary_bin_index] + bins[boundary_bin_index+1]) / 2.0
            boundary_h_bearing = relative_angle_rad / np.pi
        
        # LiDAR는 2D이므로 V_Bearing = 0.0
        boundary_feats = np.array([boundary_h_bearing, 0.0, boundary_dist_2d], dtype=np.float32)

        # (3 + 6 + 3 + 3 = 15 dims)
        obs = np.concatenate([
            leader_feats, 
            np.array(allies_feats, dtype=np.float32), 
            np.array(enemy_feats, dtype=np.float32), 
            boundary_feats
        ], axis=0).astype(np.float32)
        
        return obs


    def _do_action(self, agent, action):
        # [-1, 1] 범위의 2차원 액션을 '3D 방위각'으로 해석
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action[agent] = a.copy()
        self._current_action[agent] = a.copy()

        # 1. AI 출력을 각도(radian)로 변환
        # a[0] = 수평 방위 (-1.0 ~ 1.0) -> (-pi ~ +pi) rad
        # (드론 기준: -1.0=왼쪽, 0.0=앞, 1.0=오른쪽)
        relative_yaw_rad = float(a[0]) * np.pi 
        
        # a[1] = 수직 방위 (-1.0 ~ 1.0) -> (-pi/2 ~ +pi/2) rad
        # (드론 기준: -1.0=아래, 0.0=수평, 1.0=위)
        relative_pitch_rad = float(a[1]) * (np.pi / 2.0)

        speed = self.agent_constant_speed

        # 2. 드론 기준(Body Frame)의 3D 속도 벡터 계산 (NED 좌표계 기준)
        #    (x=앞, y=오른쪽, z=아래)
        horizontal_speed = speed * np.cos(relative_pitch_rad)
        
        v_x_rel = horizontal_speed * np.cos(relative_yaw_rad) # +X: 앞 (Forward)
        v_y_rel = horizontal_speed * np.sin(relative_yaw_rad) # +Y: 오른쪽 (Right)
        v_z_rel = speed * np.sin(relative_pitch_rad)          # +Z: 아래 (Down)
        
        # (참고) AirSim의 수직 방위각(pitch)은 '아래'가 +이므로,
        # v_z_rel 계산 시 - 부호가 필요 없습니다.
        # (a[1]=1.0 -> 위) -> pitch=-pi/2 -> sin()=-1 -> v_z_rel = -speed (위로)
        # (a[1]=-1.0 -> 아래) -> pitch=+pi/2 -> sin()=1 -> v_z_rel = +speed (아래로)

        # NumPy 배열로 변환
        v_rel_vec_np = np.array([v_x_rel, v_y_rel, v_z_rel], dtype=np.float32)

        # 3. 드론의 현재 방향(쿼터니언) 가져오기
        agent_quat = self.client.simGetVehiclePose(agent).orientation

        # 4. (NEW) 쿼터니언 -> 회전 행렬 R_wb (body → world)
        # (사용자가 제공한 _calculate_relative_bearing의 로직과 동일)
        w = agent_quat.w_val
        x = agent_quat.x_val
        y = agent_quat.y_val
        z = agent_quat.z_val

        R_wb = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
            [    2*(x*y + z*w),   1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y)]
        ], dtype=np.float32)

        # 5. (NEW) 로컬 속도 벡터를 월드 속도 벡터로 변환
        # v_world = R_wb * v_local
        v_world_vec_np = R_wb @ v_rel_vec_np

        # 6. 최종 계산된 '월드 기준' 3D 속도로 이동
        self.client.moveByVelocityAsync(
            float(v_world_vec_np[0]), # vx
            float(v_world_vec_np[1]), # vy
            float(v_world_vec_np[2]), # vz
            duration=self.dt,
            vehicle_name=agent
        )
        

    # --------------------- 보상 ---------------------
    def _compute_reward(self, agent, all_poses):
        agent_pos = all_poses[agent]
        leader_pos = all_poses["Drone1"]

        # (1) 에이전트 간 충돌
        for other in self.agents:
            if other == agent:
                continue
            other_pos = all_poses[other]
            if np.linalg.norm(agent_pos[:2] - other_pos[:2]) < 0.5:
                return -150.0, True  # 충돌 시 종료

        # (2) 리더와의 거리 계산
        rel = leader_pos - agent_pos
        dist = np.hypot(rel[0], rel[1])

        # 너무 가까움 or 너무 멀면 종료
        if dist < 0.5 or dist > 60.0:
            return -150.0, True

        # 거리 보상
        if 5.0 <= dist <= 10.0:
            reward = 3.0
        elif 0.5 <= dist < 5.0:
            reward = -150.0 + (153.0) * math.exp(-((dist - 5.0) ** 2) / (2 * 1.5 ** 2))
        elif 10.0 < dist <= 60.0:
            reward = -150.0 + (153.0) * math.exp(-((dist - 10.0) ** 2) / (2 * 15.0 ** 2))
        else:
            reward = -150.0

        if self.leader_stop:
            return -150.0, True

        return float(reward), True

    # --------------------- 팀 보상 집계 ---------------------
    def _team_reward_and_done(self, per_agent_results, mission_accomplished=False):
        # 안전 게이트: 한 명이라도 종료이면 즉시 실패
        any_fail = any(done_i for (_, done_i) in per_agent_results)
        if any_fail:
            return -200.0, True, {"final_status": "FAIL_CRASH"}

        # 미션 성공: 유인기가 목적지 도착
        if mission_accomplished:
            tracking_rewards = [r_i for (r_i, _) in per_agent_results]
            final_reward = float(np.mean(tracking_rewards)) + 500.0
            return final_reward, True, {"final_status": "SUCCESS"}

        # 미션 진행 중: 추종 보상의 평균
        tracking_rewards = [r_i for (r_i, _) in per_agent_results]
        return float(np.mean(tracking_rewards)), False, {}

    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self, t):
        # t 인자는 더 이상 안 쓰지만 호환성을 위해 남겨둠
        name = self.dynamic_name
        fixed_z = self.fixed_z
        attack_speed = 8.0 # 장애물 공격 속도
        STOP_DISTANCE = 1.0

        # 상태 변수 초기화 (reset에서 안 되었을 경우 대비)
        if not hasattr(self, "_obstacle_state"):
            self._obstacle_state = "IDLE"
            self._state_start_step = self._timestep 
            self._idle_duration = random.randint(20, 50) 
            self._idle_pos = None
            self._chase_mode = None

        # 현재 상태가 지속된 스텝 수 계산
        steps_elapsed = self._timestep - self._state_start_step

        leader_pose = self.client.simGetObjectPose("Drone1").position
        obstacle_pose = self.client.simGetObjectPose(name).position
        lx, ly, lz = leader_pose.x_val, leader_pose.y_val, leader_pose.z_val
        cx, cy, cz = obstacle_pose.x_val, obstacle_pose.y_val, obstacle_pose.z_val

        dx, dy = lx - cx, ly - cy
        dist_2d = math.sqrt(dx ** 2 + dy ** 2) + 1e-9
        
        # ------------------ IDLE 모드 ------------------
        if self._obstacle_state == "IDLE":
            # 1. 대기 시간 체크 (스텝 기준)
            if steps_elapsed > self._idle_duration:
                self._obstacle_state = "CHASE"
                self._state_start_step = self._timestep
                self._chase_mode = "STRAIGHT"
                print(f"[장애물] IDLE -> CHASE 전환 (경과: {steps_elapsed} step)")
                return

            # 2. 대기 중 움직임 (배회)
            if self._idle_pos is None:
                radius = random.uniform(60.0, 70.0)
                angle = random.uniform(0, 2 * np.pi)
                # (0,0) 기준 배회 위치 설정
                self._idle_pos = (0.0 + radius * np.cos(angle), 0.0 + radius * np.sin(angle))
            
            ix, iy = self._idle_pos
            dx_i, dy_i = ix - cx, iy - cy
            dist_idle = math.sqrt(dx_i ** 2 + dy_i ** 2)
            
            if dist_idle > 1.0:
                vx = dx_i / dist_idle * 2.0
                vy = dy_i / dist_idle * 2.0
                vz = (fixed_z - cz) * 0.5
                self.client.moveByVelocityAsync(vx, vy, vz, duration=self.dt, vehicle_name=name)
            else:
                self.client.moveByVelocityAsync(0, 0, 0, duration=self.dt, vehicle_name=name)
        
        # ------------------ CHASE 모드 ------------------
        elif self._obstacle_state == "CHASE":
            chase_limit = 150 # 약 15초(dt=0.1 기준) 동안만 추격
            
            # 1. 추격 시간 초과 시 복귀
            if steps_elapsed > chase_limit:
                self._obstacle_state = "RETURN"
                self._state_start_step = self._timestep
                self._idle_pos = None
                print(f"[장애물] 추격 시간 초과 -> RETURN")
                return

            # 2. 충돌(피격) 체크
            if dist_2d <= STOP_DISTANCE:
                self.client.moveByVelocityAsync(0, 0, 0, duration=self.dt, vehicle_name=name)
                self.leader_stop = True 
                print(f"💥[장애물] 리더 격추 성공!")
                return

            # 3. 추격 이동
            dir_x = dx / dist_2d
            dir_y = dy / dist_2d

            if self._chase_mode == "STRAIGHT":
                vx = dir_x * attack_speed
                vy = dir_y * attack_speed
            
            # Z축은 유인기 고도(lz)를 향해 이동
            vz = (lz - cz) * 2.0 
            self.client.moveByVelocityAsync(vx, vy, vz, duration=self.dt, vehicle_name=name)

        # ------------------ RETURN 모드 ------------------
        elif self._obstacle_state == "RETURN":
            if self._idle_pos is None:
                radius = random.uniform(30.0, 50.0)
                angle = random.uniform(0, 2 * np.pi)
                self._idle_pos = (0.0 + radius * np.cos(angle), 0.0 + radius * np.sin(angle))

            ix, iy = self._idle_pos
            dx_r, dy_r = ix - cx, iy - cy
            dist_return = math.sqrt(dx_r ** 2 + dy_r ** 2)
            
            if dist_return > 2.0:
                vx = dx_r / dist_return * 5.0
                vy = dy_r / dist_return * 5.0
                vz = (fixed_z - cz) * 0.5
                self.client.moveByVelocityAsync(vx, vy, vz, duration=self.dt, vehicle_name=name)
            else:
                self._obstacle_state = "IDLE"
                self._state_start_step = self._timestep
                self._idle_duration = random.randint(20, 50)
                print("[장애물] 복귀 완료 -> IDLE")

    # ======================================================================
    # PettingZoo API-ish
    # ======================================================================
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._setup_flight()
        #self._generate_leader_waypoints()
        self.current_waypoint_idx = 0
        self._teleport_to_start()

        self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
        self.client.simFlushPersistentMarkers()

        # 에피소드 시작 시 버퍼 리셋
        self._timestep = 0
        self._last_pose.clear()
        self._last_time.clear()
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}
        self._cluster_tracks = {a: {} for a in self.possible_agents}
        self._next_cluster_id = {a: 0 for a in self.possible_agents}
        self.enemy_clusters = {a: [] for a in self.possible_agents}
        self.ally_clusters = {a: [] for a in self.possible_agents}

        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1")
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent)

        now_t = 0.0
        obs_list = [self._get_obs(a, now_t, all_poses) for a in self.agents]

        self._obstacle_state = "IDLE"
        self._state_start_step = 0          # 0 스텝부터 시작
        self._idle_duration = random.randint(20, 50) 
        self._idle_pos = None
        self._chase_mode = None
        self.leader_stop = False
        
        return obs_list

    def step(self, actions):
        self._timestep += 1
        t = time.time()
        self._update_dynamic_obstacle(t)
        now_t = self._timestep * self.dt

        # A) LiDAR 기반 트랙 갱신
        for agent in self.agents:
            if agent.startswith("Follower"):
                self._update_clusters_and_tracks(agent, now_t)

        # 디버깅 출력
        #self._debug_print_clusters()

        # B) 액션 적용
        for agent, act in zip(self.agents, actions):            
            self._do_action(agent, act)

        # C) 리더 이동/시각화
        mission_accomplished = self._update_leader_movement()

        # 모든 위치 정보를 이 시점에 "한 번만" 가져옴
        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1")
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent)

        # D) 관측/보상/종료 신호
        obs_list, per_agent_results, per_agent_infos = [], [], []
        for agent in self.agents:
            o = self._get_obs(agent, now_t, all_poses)
            r, done_i = self._compute_reward(agent, all_poses)

            obs_list.append(o)
            per_agent_results.append((float(r), bool(done_i)))
            per_agent_infos.append({"reward": float(r)})

        team_reward, done_all, final_team_info = self._team_reward_and_done(per_agent_results, mission_accomplished)

        n = len(self.agents)
        rewards_list = [team_reward] * n
        dones_list = [done_all] * n
        infos_list = []
        for i in range(n):
            info_i = per_agent_infos[i].copy()
            info_i.update(final_team_info)
            infos_list.append(info_i)

        # 가디언 로직(리더 피격/차폐)은 현재 주석 처리 상태
        return obs_list, rewards_list, dones_list, infos_list
