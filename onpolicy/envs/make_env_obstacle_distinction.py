# from MARL_test import ParallelEnv
import gym
import numpy as np
import airsim
import math
import time
import random
from sklearn.cluster import DBSCAN
from collections import deque


class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",                 # 기본값
        follower_names=("Follower0", "Follower1", "Follower2"),
        dynamic_name = ("DynamicObstacle0",),
        lidar_name="LidarSensor",               # 라이다 공통명 (실제 센서는 f"{agent}_{lidar_name}")
        min_samples=5,
        step_length=0.01,
        fixed_z=-15.0,
        leader_velocity=1.0,                    # 유인기 속도(m/s) 파라미터화
        optimal_distance=10.0,                  # 추종 최적 거리(시각화용 링)
        far_cutoff=60.0,                        # 이탈 종료 거리(시각화용 링)
        too_close=0.5,                          # 유인기와 최소 거리
        dt=0.01,                                # 내부 시뮬레이션 타임스텝(초)
        do_visualize=True                       # 원/포인트 시각화 on/off
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]
        self.dynamic_agents = list(dynamic_name)
        # 라이다 / DBSCAN
        self.lidar_name = lidar_name
        self.min_samples = int(min_samples)
        self.eps = 0.3

        # 관측공간(동적 객체 K-NN)
        self.K_nearest = 3                  # K-최근접 객체 개수
        self.match_dist_max = 2.0           # 프레임 간 클러스터 매칭 허용 거리[m]
        self._cluster_tracks = {a: {} for a in self.possible_agents}  # agent별 트랙 사전
        self._next_cluster_id = {a: 0 for a in self.possible_agents}  # agent별 ID 증가기
        self.enemy_clusters = {a: [] for a in self.possible_agents}   # 적군 클러스터 리스트
        self.ally_clusters = {a: [] for a in self.possible_agents}    # 아군 클러스터 리스트
        self.cluster_N_stack = 50            # 클러스터 centroid 히스토리 길이

        # 거리 기반 피아식별 파라미터
        self.friend_radius = 45.0      # 리더로부터 45m 이내 → 아군 후보
        self.max_lidar_radius = 80.0   # 80m 이내 포인트만 피아 후보로 사용

        # 에피소드당 고정할 트랙 ID (아군/적군 3개씩)
        self.ally_track_ids = {a: [None] * self.K_nearest for a in self.possible_agents}
        self.enemy_track_ids = {a: [None] * self.K_nearest for a in self.possible_agents}

        # 에이전트/리더 속도 산출용 버퍼 (자기 속도만 관측에 사용)
        self._last_pose = {}     # {"name": np.array([x,y,z])}
        self._last_time = {}     # {"name": t_float}
        self.vmax_self = 2.0     # 자기 속도 상한(m/s) (학습 안정용)

        self._timestep = 1.0

        # 액션 버퍼
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}

        # 이동 모드: 텔레포트 대신 moveByVelocity 기반
        self.use_teleport = False  # False면 위치 차분으로 속도 추정

        # 클라이언트
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = float(fixed_z)
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)

        # 액션 → 속도[m/s] 변환 스케일 (기본 최대 속도 스케일)
        self.max_cmd_speed = self.step_length / self.dt  # 예: 0.05 / 0.01 = 5 m/s

        self.leader_velocity = float(leader_velocity)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)

        self._first_setup = True
        self.leader_stop = False

        # ===== obs / act / share_obs spaces =====
        # obs 구조:
        #  self_vel(3) +
        #  rel_to_leader: bearing, dist (2) +
        #  ally K개:   [bearing, dist] * K +
        #  enemy K개:  [bearing, dist] * K
        self.num_ally = self.K_nearest
        self.num_enemy = self.K_nearest

        obs_dim = 3 + 2 + 2 * self.num_ally + 2 * self.num_enemy  # K=3이면 17
        share_obs_dim = obs_dim * len(self.possible_agents)

        # 관측 bound 설정
        low_self = [-20.0, -20.0, -20.0]
        high_self = [20.0, 20.0, 20.0]

        low_bearing = -1.0
        high_bearing = 1.0
        low_dist = 0.0
        high_dist = 200.0

        per_agent_low = []
        per_agent_high = []

        # 1) self_vel
        per_agent_low += low_self
        per_agent_high += high_self

        # 2) 리더에 대한 bearing + dist
        per_agent_low += [low_bearing, low_dist]
        per_agent_high += [high_bearing, high_dist]

        # 3) 아군 K개
        for _ in range(self.num_ally):
            per_agent_low += [low_bearing, low_dist]
            per_agent_high += [high_bearing, high_dist]

        # 4) 적군 K개
        for _ in range(self.num_enemy):
            per_agent_low += [low_bearing, low_dist]
            per_agent_high += [high_bearing, high_dist]

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low, dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,), dtype=np.float32
            ) for agent in self.possible_agents
        }

        # --- 액션 스페이스: yaw 이산 선택 ---
        self.min_speed = 0.0   # 최소 속도 (m/s)
        self.max_speed = 10.0   # 최대 속도 (m/s)
        self.action_spaces = {
            agent: gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,), 
                dtype=np.float32
            ) for agent in self.possible_agents
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

        # 디버깅 플래그
        self.debug_clusters = True

    # ======================================================================
    # 쿼터니언 → 회전행렬 (world_from_body)
    # ======================================================================
    def _quat_to_rot(self, q):
        """
        AirSim quaternion (w,x,y,z) -> 3x3 회전행렬 (world_from_body)
        """
        w = q.w_val
        x = q.x_val
        y = q.y_val
        z = q.z_val

        ww, xx, yy, zz = w*w, x*x, y*y, z*z

        R = np.array([
            [ww + xx - yy - zz,     2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),         ww - xx + yy - zz,   2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),       ww - xx - yy + zz]
        ], dtype=np.float32)

        return R

    # ======================================================================
    # 라이다 데이터 → 월드좌표 포인트
    # ======================================================================
    def _lidar_sensor_name(self, agent: str) -> str:
        # 실제 AirSim 설정: 각 드론에 "Follower0_LidarSensor" 식으로 붙어 있다고 가정
        return self.lidar_name

    def _LidarPointsWorld(self, agent):
        """
        라이다 포인트를 '에이전트 로컬 좌표'에서 '월드 좌표'로 변환해서 리턴.
        (라이다가 기체에 고정되어 있다고 가정)
        """
        specific_lidar_name = f"{agent}_{self.lidar_name}"
        
        try:
            ld = self.client.getLidarData(
                lidar_name=specific_lidar_name,
                vehicle_name=agent
            )
        except Exception as e:
            # 혹시라도 이름이 틀렸을 경우를 대비해 디버깅 메시지 출력
            print(f"Error getting LiDAR for {agent} with name '{specific_lidar_name}': {e}")
            return np.empty((0, 3), dtype=np.float32)
        arr = np.array(ld.point_cloud, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        pts_local = arr.reshape(-1, 3)  # body frame (기체 기준)

        # 에이전트의 월드 포즈
        pose = self.client.simGetVehiclePose(vehicle_name=agent)
        pos_w = np.array(
            [pose.position.x_val,
             pose.position.y_val,
             pose.position.z_val],
            dtype=np.float32
        )
        R_wb = self._quat_to_rot(pose.orientation)  # world_from_body

        # 로컬 → 월드
        pts_world = (R_wb @ pts_local.T).T + pos_w  # (N,3)

        return pts_world

    # ======================================================================
    # 포즈/속도 관련 헬퍼
    # ======================================================================
    def _get_pose_xyz(self, name):
        pose = self.client.simGetObjectPose(object_name=name).position
        return np.array([pose.x_val, pose.y_val, pose.z_val], dtype=np.float32)

    def _get_yaw(self, name: str) -> float:
        """
        드론의 현재 yaw(머리 방향) [rad] 반환
        """
        pose = self.client.simGetVehiclePose(vehicle_name=name)
        _, _, yaw = airsim.to_eularian_angles(pose.orientation)
        return float(yaw)

    def _bearing_and_distance(self, src_pos, src_yaw, tgt_pos):
        """
        src_pos: np.array([x,y,z])
        src_yaw: [rad]
        tgt_pos: np.array([x,y,z])

        return:
            bearing_norm: [-1, 1]   ( = (angle_world - yaw)/π )
            distance: [m]
        """
        dx = float(tgt_pos[0] - src_pos[0])
        dy = float(tgt_pos[1] - src_pos[1])
        dist = math.hypot(dx, dy)

        angle_world = math.atan2(dy, dx)
        bearing = angle_world - src_yaw

        # [-π, π] 로 wrap
        while bearing > math.pi:
            bearing -= 2.0 * math.pi
        while bearing < -math.pi:
            bearing += 2.0 * math.pi

        bearing_norm = bearing / math.pi  # [-1, 1]
        return bearing_norm, dist

    def _clusters_to_polar_feats(self, agent, cluster_list, K, yaw, pos):
        """
        cluster_list: self.ally_clusters[agent] 또는 self.enemy_clusters[agent]
        K: 최대 몇 개까지 관측에 넣을지
        yaw: 에이전트 yaw [rad]
        pos: 에이전트 위치 np.array([x,y,z])
        """
        feats = []
        for i in range(K):
            if i < len(cluster_list):
                cpos = cluster_list[i]["centroid"]
                bearing, dist = self._bearing_and_distance(pos, yaw, cpos)
                feats.extend([bearing, dist])
            else:
                feats.extend([0.0, 0.0])  # 패딩
        return np.array(feats, dtype=np.float32)

    # ======================================================================
    # 라이다 데이터 → DBSCAN → 트랙 업데이트 (거리 기반 피아식별)
    # ======================================================================
    def _update_clusters_and_tracks(self, agent, now_t):
        # 0) 현재 포인트맵 (world)
        pts_w = self._LidarPointsWorld(agent)

        # =========================
        # 1) DBSCAN 클러스터링
        # =========================
        if pts_w.shape[0] == 0:
            clusters = []
        else:
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pts_w)
            labels = db.labels_
            clusters = []
            for cid in np.unique(labels):
                if cid == -1:
                    continue
                idx = (labels == cid)
                cpts = pts_w[idx]
                centroid = cpts.mean(axis=0)

                clusters.append({
                    "centroid": centroid,
                    "count": int(cpts.shape[0]),
                })

        # =========================
        # 2) 트랙 업데이트 (예측 + 최근접 매칭)
        # =========================
        tracks = self._cluster_tracks[agent]

        cur_centroids = [c["centroid"] for c in clusters]
        cur_used = [False] * len(cur_centroids)

        # ---- 예측 위치 계산용 헬퍼 (history 이용) ----
        def _predict_pos(track_state):
            """
            track_state['history'] 에 쌓인 (t, pos)들을 이용해
            평균 속도로 앞으로 약간 예측한 위치를 반환.
            """
            hist = track_state.get("history", None)
            if hist is None or len(hist) < 2:
                return track_state["centroid"]

            t0, p0 = hist[0]
            t_last, p_last = hist[-1]
            dt_hist = max(1e-6, t_last - t0)
            v_hist = (p_last - p0) / dt_hist
            dt_future = now_t - t_last
            pred_pos = p_last + v_hist * dt_future
            return pred_pos

        # ---- 2-1) 기존 트랙에 새 클러스터 매칭 ----
        for tid, st in list(tracks.items()):
            best_j, best_d = -1, 1e9

            pred_pos = _predict_pos(st)

            for j, cen in enumerate(cur_centroids):
                if cur_used[j]:
                    continue
                d = np.linalg.norm(cen - pred_pos)
                if d < best_d:
                    best_d, best_j = d, j

            if best_j >= 0 and best_d <= self.match_dist_max:
                # 매칭 성공
                new_c = cur_centroids[best_j]

                # history 업데이트
                hist = st.get("history", None)
                if hist is None:
                    hist = deque(maxlen=self.cluster_N_stack)
                    hist.append((st["last_t"], st["centroid"].copy()))
                hist.append((now_t, new_c.copy()))

                tracks[tid] = {
                    "centroid": new_c.copy(),
                    "last_t": now_t,
                    "history": hist,
                }
                cur_used[best_j] = True
            else:
                # 오래 안 갱신된 트랙 삭제
                if now_t - st["last_t"] > 1.0:
                    del tracks[tid]

        # ---- 2-2) 매칭 안 된 클러스터 = 신규 트랙 생성 ----
        for j, cen in enumerate(cur_centroids):
            if not cur_used[j]:
                tid = self._next_cluster_id[agent]
                self._next_cluster_id[agent] += 1

                hist = deque(maxlen=self.cluster_N_stack)
                hist.append((now_t, cen.copy()))

                tracks[tid] = {
                    "centroid": cen.copy(),
                    "last_t": now_t,
                    "history": hist,
                }

        # =========================
        # 3) 고정된 트랙 ID 기반으로 아군/적군 리스트 갱신
        # =========================
        if hasattr(self, "ally_track_ids"):
            self._refresh_fixed_cluster_lists(agent)

    def _refresh_fixed_cluster_lists(self, agent):
        """
        self.ally_track_ids / enemy_track_ids 에 저장된 track id 순서를 기준으로
        self.ally_clusters / self.enemy_clusters 내용을 갱신.
        (레이블/순서는 에피소드 동안 고정)
        """
        tracks = self._cluster_tracks[agent]

        ally_list = []
        for tid in self.ally_track_ids.get(agent, []):
            if tid is None:
                continue
            st = tracks.get(tid)
            if st is None:
                continue
            ally_list.append({
                "id": tid,
                "centroid": st["centroid"].copy(),
            })

        enemy_list = []
        for tid in self.enemy_track_ids.get(agent, []):
            if tid is None:
                continue
            st = tracks.get(tid)
            if st is None:
                continue
            enemy_list.append({
                "id": tid,
                "centroid": st["centroid"].copy(),
            })

        self.ally_clusters[agent] = ally_list
        self.enemy_clusters[agent] = enemy_list
        # ======================================================================
    # 디버깅용: 아군/적군 클러스터 출력
    # ======================================================================
    def _debug_print_clusters(self):
        """
        각 에이전트 기준으로 ally_clusters / enemy_clusters에
        어떤 클러스터(id, 위치)가 들어가 있는지 출력해서
        거리 기반 피아식별이 잘 되는지 확인하기 위한 디버깅 함수.
        """
        if not self.debug_clusters:
            return

        # 너무 자주 안 찍히게, N 스텝마다만 찍기
        if self._timestep % 10 != 0:
            return

        print(f"\n[CLUSTER DEBUG][step {self._timestep}]")
        print(f"  friend_radius = {self.friend_radius:.1f} m, "
              f"max_lidar_radius = {self.max_lidar_radius:.1f} m")

        # 리더 위치 (피아식별 기준이 되는 중심)
        leader_pos = self._get_pose_xyz("Drone1")

        for agent in self.agents:
            allies = self.ally_clusters.get(agent, [])
            enemies = self.enemy_clusters.get(agent, [])

            print(f"\n▶ agent = {agent}")
            print(f"    allies = {len(allies)}, enemies = {len(enemies)}")

            # 아군 클러스터들
            for a in allies:
                c = a["centroid"]
                d_leader = np.linalg.norm((c - leader_pos)[:2])
                print(
                    f"    [ALLY ] id={a['id']}, "
                    f"d_leader={d_leader:.1f} m, "
                    f"pos=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"
                )

            # 적군 클러스터들
            for e in enemies:
                c = e["centroid"]
                d_leader = np.linalg.norm((c - leader_pos)[:2])
                print(
                    f"    [ENEMY] id={e['id']}, "
                    f"d_leader={d_leader:.1f} m, "
                    f"pos=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})"
                )

        # 각 에이전트별로 고정된 트랙 ID도 같이 확인하고 싶으면 아래 주석 해제
        # for agent in self.agents:
        #     print(f"\n  fixed IDs for {agent}:")
        #     print(f"    ally_track_ids  = {self.ally_track_ids[agent]}")
        #     print(f"    enemy_track_ids = {self.enemy_track_ids[agent]}")


    def _initialize_fixed_clusters(self, agent, now_t):
        """
        에피소드 시작 시 1번만 실행:
        - 현재 LiDAR 포인트로 트랙 생성
        - 리더와의 거리 기준으로 아군/적군 후보 분류
        - 각 3개씩 track id 고정
        """
        # 먼저 현재 프레임에서 트랙 생성/업데이트
        self._update_clusters_and_tracks(agent, now_t)

        tracks = self._cluster_tracks[agent]
        leader_pos = self._get_pose_xyz("Drone1")

        ally_candidates = []
        enemy_candidates = []

        for tid, st in tracks.items():
            c = st["centroid"]
            # 리더 기준 2D 거리
            d = np.linalg.norm((c - leader_pos)[:2])

            if d <= self.friend_radius:
                ally_candidates.append((d, tid))
            elif d <= self.max_lidar_radius:
                enemy_candidates.append((d, tid))

        ally_candidates.sort(key=lambda x: x[0])   # 가까운 순
        enemy_candidates.sort(key=lambda x: x[0])  # 가까운 순

        ally_ids = [tid for (d, tid) in ally_candidates[:self.K_nearest]]
        enemy_ids = [tid for (d, tid) in enemy_candidates[:self.K_nearest]]

        # 3개가 안 되면 None으로 패딩
        while len(ally_ids) < self.K_nearest:
            ally_ids.append(None)
        while len(enemy_ids) < self.K_nearest:
            enemy_ids.append(None)

        self.ally_track_ids[agent] = ally_ids
        self.enemy_track_ids[agent] = enemy_ids

        # 실제 클러스터 리스트도 한 번 생성 (관측용 구조)
        self._refresh_fixed_cluster_lists(agent)

    # ======================================================================
    # 속도 추정 (자기 속도만 관측에 사용)
    # ======================================================================
    def _get_self_velocity(self, name, now_t, current_pos):
        """
        [최적화됨] _get_pose_xyz() API 호출을 제거하고,
        미리 계산된 current_pos를 인자로 받습니다.
        """
        if self.use_teleport and name in self._last_action:
            a = self._last_action[name]
            vx = float(a[0]) * float(self.step_length) / self.dt
            vy = float(a[1]) * float(self.step_length) / self.dt
            vz = 0.0
            v = np.array([vx, vy, vz], dtype=np.float32)
        else:
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
    def _setup_flight(self):
        if self._first_setup:
            self.client.reset()
            
            # 리더, 팔로워, 장애물 전체 리스트
            all_vehicles = ["Drone1"] + self.possible_agents + self.dynamic_agents
            
            # 1. 시동 걸기 (Arming)
            for v_name in all_vehicles:
                self.client.enableApiControl(True, vehicle_name=v_name)
                self.client.armDisarm(True, vehicle_name=v_name)

            # 2. 초기 위치로 텔레포트 (Takeoff 생략)
            self._teleport_to_start()
            
            # 3. [핵심] 텔레포트 직후 "강제 호버링" 명령 전송
            # 이 코드가 없으면 텔레포트 하자마자 바닥으로 추락합니다.
            for v_name in all_vehicles:
                self.client.moveByVelocityZAsync(
                    vx=0, 
                    vy=0, 
                    z=float(self.fixed_z), # 목표 고도 유지
                    duration=3.0,          # 충분한 시간 동안 자세 잡기
                    vehicle_name=v_name
                )
            
            # 4. 물리 엔진이 안정화될 때까지 대기
            time.sleep(2.0)
            
            self._first_setup = False


    def _teleport_to_start(self):
        start_cfg = {
            "Drone1": (5.0, 2.5, self.fixed_z),
            "Follower0": (0.0, 0.0, self.fixed_z),
            "Follower1": (0.0, 2.5, self.fixed_z),
            "Follower2": (0.0, 5.0, self.fixed_z),
            "DynamicObstacle0" : (0.0, 50.0, self.fixed_z),
        }

        self.client.enableApiControl(True, vehicle_name="Drone1")
        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
        for agent in self.dynamic_agents:
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

    # ======================================================================
    # 유인기 경로/이동/시각화
    # ======================================================================
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
    # 보상 관련 헬퍼
    # ======================================================================
    def _formation_reward(self, agent_pos, leader_pos):
        """
        유인기와의 거리 ~10m 유지용 포메이션 보상
        """
        rel = leader_pos - agent_pos
        dist = math.hypot(float(rel[0]), float(rel[1]))

        # 너무 극단적인 경우에는 추가 패널티
        if dist < 0.5 or dist > 60.0:
            return -5.0

        ideal = 10.0
        sigma = 10.0
        r = 3.0 * math.exp(-((dist - ideal) ** 2) / (2.0 * sigma ** 2)) - 1.0
        # 대략 [-1, +2] 정도
        return r

    def _guardian_reward(self, agent_pos, leader_pos, dynamic_pos):
        """
        유인기-장애물 사이에서 방패 역할을 잘 할수록 +보상
        """
        d_lo = np.linalg.norm(leader_pos[:2] - dynamic_pos[:2])  # 리더-장애물
        d_ao = np.linalg.norm(agent_pos[:2] - dynamic_pos[:2])   # 에이전트-장애물

        ALERT_DIST = 80.0
        if d_lo > ALERT_DIST:
            # 장애물이 너무 멀면 아직 위협 아님
            return 0.0

        if d_ao < d_lo:
            # 장애물보다 유인기 쪽에서 막고 있는 상태
            score = (d_lo - d_ao) / max(d_lo, 1e-3)  # 0~1
            return 2.0 * score   # 최대 +2 정도
        else:
            # 장애물 뒤에 숨어 있으면 약한 패널티
            return -0.5

    def _check_intercept(self, all_poses, dynamic_pos):
        """
        장애물이 팔로워 드론에 먼저 '닿았는지' 체크 (요격 성공 이벤트)
        """
        for agent in self.agents:
            agent_pos = all_poses[agent]
            d = np.linalg.norm(agent_pos[:2] - dynamic_pos[:2])
            if d < 1.0:   # 요격 거리 기준
                return agent
        return None

    def _compute_reward(self, agent, all_poses, dynamic_pos):
        agent_pos = all_poses[agent]
        leader_pos = all_poses["Drone1"]
        if np.linalg.norm(agent_pos[:2] - leader_pos[:2]) < 1.0:
            return -200.0, True

        # 1) 에이전트끼리 충돌 → 큰 패널티 + 에피소드 종료
        for other in self.agents:
            if other == agent:
                continue
            other_pos = all_poses[other]
            if np.linalg.norm(agent_pos[:2] - other_pos[:2]) < 0.5:
                return -150.0, True  # 충돌 시 종료

        # 2) 포메이션 보상
        r_form = self._formation_reward(agent_pos, leader_pos)

        # 3) 가디언 위치 보상
        r_guard = self._guardian_reward(agent_pos, leader_pos, dynamic_pos)

        # 4) 리더가 이미 장애물에 맞아 멈춘 상태라면 패널티 주고 종료
        if self.leader_stop:
            return -150.0, True

        r_total = r_form + r_guard
        return float(r_total), False

    def _team_reward_and_done(
        self,
        per_agent_results,
        mission_accomplished=False,
        intercepted_agent=None,
        hit_leader=False,
    ):
        """
        팀 레벨 보상/종료:
          - hit_leader: 장애물이 유인기에 충돌 → 큰 실패
          - intercepted_agent: 팔로워가 먼저 요격 → 큰 성공
          - any_fail: 에이전트끼리 충돌 등
          - mission_accomplished: 유인기가 목적지 도달
        """
        any_fail = any(done_i for (_, done_i) in per_agent_results)

        if hit_leader:
            # 장애물이 유인기에 닿음 → 최악의 실패
            return -800.0, True, {"final_status": "FAIL_HIT_LEADER"}

        if intercepted_agent is not None:
            # 장애물이 팔로워 드론에 먼저 부딪힘 → 요격 성공
            tracking_rewards = [r_i for (r_i, _) in per_agent_results]
            base = float(np.mean(tracking_rewards))
            return base + 500.0, True, {
                "final_status": "SUCCESS_INTERCEPT",
                "interceptor": intercepted_agent,
            }

        if any_fail:
            return -200.0, True, {"final_status": "FAIL_CRASH"}

        if mission_accomplished:
            tracking_rewards = [r_i for (r_i, _) in per_agent_results]
            final_reward = float(np.mean(tracking_rewards)) + 300.0
            return final_reward, True, {"final_status": "SUCCESS_REACH_GOAL"}

        tracking_rewards = [r_i for (r_i, _) in per_agent_results]
        return float(np.mean(tracking_rewards)), False, {}
    
    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self):
        # 1. 리더 위치 조회
        leader_start_x, leader_start_y = 5.0, 2.5
        
        attack_speed = 3.0  # 속도
        safe_duration = self.dt * 2.0 # 끊김 방지 (중요!)
        STOP_DISTANCE = 2.0

        # 2. 모든 장애물에 대해 반복 (중요!)
        for name in self.dynamic_agents:
            # 장애물 위치 조회
            obstacle_pose = self.client.simGetObjectPose(object_name=name).position
            cx, cy = obstacle_pose.x_val, obstacle_pose.y_val

            # 방향 벡터 계산
            dx = leader_start_x - cx
            dy = leader_start_y - cy
            dist_2d = math.sqrt(dx*dx + dy*dy) + 1e-6

            # 속도 벡터 계산
            vx = (dx / dist_2d) * attack_speed
            vy = (dy / dist_2d) * attack_speed
            
            # [사용자 코드가 좋았던 점 적용]
            # moveByVelocityZAsync를 사용하여 고도(fixed_z)를 꽉 잡아줍니다.
            self.client.moveByVelocityZAsync(
                vx=float(vx), 
                vy=float(vy), 
                z=float(self.fixed_z), 
                duration=safe_duration, 
                vehicle_name=name
            )

            # 충돌 체크
            if dist_2d <= STOP_DISTANCE:
                print(f"💥 리더 피격 당함! ({name})")
                self.leader_stop = True
                # 충돌 시 정지 명령
                self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=name)


    def _teleport_obstacle_around_leader(self):
        # 1) 유인기 위치
        leader = self._get_pose_xyz("Drone1")
        lx, ly, lz = leader

        # 2) 반경 80~90m 랜덤 위치
        radius = random.uniform(80.0, 90.0)
        angle  = random.uniform(0, 2*np.pi)

        ox = lx + radius * math.cos(angle)
        oy = ly + radius * math.sin(angle)
        oz = self.fixed_z

        # 3) 동적 장애물만 순간이동
        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(ox, oy, oz),
                airsim.Quaternionr(0,0,0,1)
            ),
            ignore_collision=True,
            vehicle_name=self.dynamic_name
        )

        # 4) 바로 정지
        self.client.moveByVelocityAsync(0,0,0, duration=0.1, vehicle_name=self.dynamic_name)
    
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

    # ======================================================================
    # PettingZoo API-ish
    # ======================================================================
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._setup_flight()
        self._generate_leader_waypoints()
        self.current_waypoint_idx = 0
        #self.client.enableApiControl(True, vehicle_name=self.dynamic_agents)

        self._timestep = 0
        self._last_pose.clear()
        self._last_time.clear()
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        # 트랙/클러스터/ID 초기화
        self._cluster_tracks = {a: {} for a in self.possible_agents}
        self._next_cluster_id = {a: 0 for a in self.possible_agents}
        self.enemy_clusters = {a: [] for a in self.possible_agents}
        self.ally_clusters = {a: [] for a in self.possible_agents}
        # 에피소드마다 고정되는 아군/적군 클러스터 ID
        self.ally_track_ids = {a: [None] * self.K_nearest for a in self.possible_agents}
        self.enemy_track_ids = {a: [None] * self.K_nearest for a in self.possible_agents}

        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1")
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent)

        now_t = 0.0
        # 에피소드 시작 시 1회: 각 팔로워 기준으로 아군/적군 클러스터 3개씩 고정
        for agent in self.agents:
            if agent.startswith("Follower"):
                self._initialize_fixed_clusters(agent, now_t)

        obs_list = [self._get_obs(a, now_t, all_poses) for a in self.agents]

        self._episode_rewards = {a: 0.0 for a in self.possible_agents}

        self._obstacle_state = "IDLE"
        self._idle_pos = None
        self._next_chase_time = time.time() + random.uniform(1.0, 3.0)
        self._chase_mode = None
        self.leader_stop = False

        return obs_list

    def _get_obs(self, agent, now_t, all_poses):
        now_t = self._timestep * self.dt

        agent_pos = all_poses[agent]
        leader_pos = all_poses["Drone1"]

        self_vel = self._get_self_velocity(agent, now_t, agent_pos)
        yaw = self._get_yaw(agent)

        bearing_leader, dist_leader = self._bearing_and_distance(agent_pos, yaw, leader_pos)

        ally_feats = self._clusters_to_polar_feats(
            agent,
            self.ally_clusters.get(agent, []),
            self.num_ally,
            yaw,
            agent_pos
        )
        enemy_feats = self._clusters_to_polar_feats(
            agent,
            self.enemy_clusters.get(agent, []),
            self.num_enemy,
            yaw,
            agent_pos
        )

        obs = np.concatenate(
            [
                self_vel.astype(np.float32),
                np.array([bearing_leader, dist_leader], dtype=np.float32),
                ally_feats,
                enemy_feats,
            ],
            axis=0
        ).astype(np.float32)

        # ======= 🔍 디버그 출력 (상대 방위각 / 상대 거리) =======
        if self.debug_clusters and self._timestep % 5 == 0:
            print(f"\n[OBS DEBUG][step {self._timestep}] agent={agent}")
            print(f"  leader: bearing={bearing_leader:.3f}, dist={dist_leader:.2f}")

            # ally_feats: [b0, d0, b1, d1, ...]
            for i in range(self.num_ally):
                b = float(ally_feats[2*i])
                d = float(ally_feats[2*i + 1])
                print(f"  ally[{i}]: bearing={b:.3f}, dist={d:.2f}")

            # enemy_feats: [b0, d0, b1, d1, ...]
            for i in range(self.num_enemy):
                b = float(enemy_feats[2*i])
                d = float(enemy_feats[2*i + 1])
                print(f"  enemy[{i}]: bearing={b:.3f}, dist={d:.2f}")

        return obs

    def _do_action(self, agent, action):
        # action: [yaw_control(-1~1), speed_control(-1~1)]
        
        raw_yaw = float(action[0])
        raw_speed = float(action[1])

        # 1. Yaw Rate (회전 속도) 계산
        max_yaw_rate_deg = 90.0
        yaw_rate_val = math.radians(max_yaw_rate_deg) * raw_yaw 

        # 2. Forward Speed (전진 속도) 계산
        target_speed = (raw_speed + 1.0) / 2.0 * (self.max_speed - self.min_speed) + self.min_speed
        
        # 3. 현재 Yaw 각도를 가져와서 속도 벡터(Vx, Vy) 분해
        pose = self.client.simGetVehiclePose(vehicle_name=agent)
        _, _, current_yaw = airsim.to_eularian_angles(pose.orientation)
        
        vx = target_speed * math.cos(current_yaw)
        vy = target_speed * math.sin(current_yaw)

        # 4. 명령 전송 (유인기와 동일한 moveByVelocityZAsync 사용)
        # ★ 핵심: duration을 dt보다 여유 있게(1.5배~2배) 주어 명령 끊김 방지
        safe_duration = self.dt * 2.0 
        
        self.client.moveByVelocityZAsync(
            vx=vx,
            vy=vy,
            z=float(self.fixed_z), # 유인기처럼 목표 고도 고정
            duration=safe_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate_val)),
            vehicle_name=agent
        )

    def step(self, actions):
        if (self._timestep == 0):
            self._teleport_to_start()
        
        self._timestep += 1
        self._update_dynamic_obstacle()
        now_t = self._timestep * self.dt

        # ================================
        #  (0) AirSim 충돌 이벤트 검사 (가장 먼저 처리)
        # ================================
        for agent in self.agents:
            col = self.client.simGetCollisionInfo(vehicle_name=agent)

            if col.has_collided:
                other = col.object_name

                # --- A) 에이전트 ↔ 에이전트 ---
                if other in self.agents:
                    print(f"💥[에이전트 충돌] {agent} ↔ {other}")
                    return self._end_episode(-1000.0, "agent_collision")

                # --- B) 에이전트 ↔ Leader(Drone1) ---
                if other == "Drone1":
                    print(f"💥[리더 충돌] {agent}이 Drone1에 부딪힘 → 전체 실패")
                    return self._end_episode(-1000.0, "agent_hit_leader")

                # --- C) 에이전트 ↔ 동적 장애물 (Guardian 성공) ---
                if other == self.dynamic_name:
                    print(f"🛡️[가디언] {agent}이 동적장애물({self.dynamic_name})을 막음!")
                    self._teleport_obstacle_around_leader()
                    return self._end_episode(+1000.0, "guardian_block", done=False)

        # 1) 클러스터 / 트랙 업데이트
        for agent in self.agents:
            if agent.startswith("Follower"):
                self._update_clusters_and_tracks(agent, now_t)

        # 2) 에이전트 액션 적용
        for agent, act in zip(self.agents, actions):
            self._do_action(agent, act)

        # 3) 유인기 이동
        mission_accomplished = self._update_leader_movement()

        # 4) 위치들 수집
        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1")
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent)
        dynamic_pos = self._get_pose_xyz(self.dynamic_name)

        # 5) 요격 / 리더 피격 여부 체크
        intercepted_agent = self._check_intercept(all_poses, dynamic_pos)
        hit_leader = self.leader_stop

        # 6) 관측 / 개별 보상
        obs_list, per_agent_results, per_agent_infos = [], [], []
        for agent in self.agents:
            o = self._get_obs(agent, now_t, all_poses)
            r, done_i = self._compute_reward(agent, all_poses, dynamic_pos)

            obs_list.append(o)
            per_agent_results.append((float(r), bool(done_i)))
            per_agent_infos.append({"reward": float(r)})

        # 7) 팀 보상 / 종료 판정
        team_reward, done_all, final_team_info = self._team_reward_and_done(
            per_agent_results,
            mission_accomplished=mission_accomplished,
            intercepted_agent=intercepted_agent,
            hit_leader=hit_leader,
        )

        n = len(self.agents)
        rewards_list = [team_reward] * n
        dones_list = [done_all] * n
        infos_list = []
        for i, agent in enumerate(self.agents):
            # 1. 이번 스텝의 보상을 누적
            r = rewards_list[i]
            self._episode_rewards[agent] += r
            
            # 2. info 생성 (기존 로직 + 누적 보상 정보 추가)
            info_i = per_agent_infos[i].copy()
            info_i.update(final_team_info)
            
            # 에피소드가 끝났다면 info에 총 보상을 담아줍니다 (학습 로그용)
            if done_all:
                info_i["episode_reward"] = self._episode_rewards[agent]
            
            infos_list.append(info_i)

        # 3. 디버깅용 출력 (에피소드 종료 시에만)
        if done_all:
            # 모든 에이전트가 팀 보상을 공유하므로, 대표(첫 번째) 값만 출력해도 됩니다.
            # 만약 개별 보상을 쓴다면 전체 딕셔너리를 출력하세요.
            total_score = self._episode_rewards[self.agents[0]]
            print(f"\n🏁 [에피소드 종료] Total Reward: {total_score:.2f}")
            print(f"   종료 요인: {final_team_info.get('final_status', 'Unknown')}\n")

        return obs_list, rewards_list, dones_list, infos_list
