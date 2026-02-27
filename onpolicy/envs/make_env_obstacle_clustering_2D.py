# from MARL_test import ParallelEnv
import gym
import numpy as np
import airsim
import math
import time
import random
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import deque


class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",                 # 기본값
        follower_names=("Follower0", "Follower1", "Follower2"),
        lidar_name="LidarSensor",               # 라이다 공통명 (실제 센서는 f"{agent}_{lidar_name}")
        min_samples=5,
        step_length=0.01,
        fixed_z=-10.0,
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

        # 라이다 / DBSCAN
        self.lidar_name = lidar_name
        self.min_samples = int(min_samples)
        self.eps = 0.3
        self.min_pts = 5

        # 관측공간(동적 객체 K-NN)
        self.K_nearest = 3                  # K-최근접 동적 객체 개수
        self.match_dist_max = 2.0           # 프레임 간 클러스터 매칭 허용 거리[m]
        self._cluster_tracks = {a: {} for a in self.possible_agents}  # agent별 트랙 사전
        self._next_cluster_id = {a: 0 for a in self.possible_agents}  # agent별 ID 증가기
        self.enemy_clusters = {a: [] for a in self.possible_agents}   # 적군 클러스터 리스트
        self.ally_clusters = {a: [] for a in self.possible_agents}    # 아군 클러스터 리스트
        self.cluster_N_stack = 5            # 클러스터 centroid 히스토리 길이

        # 포인트맵 히스토리 (논문식 dynamic 판정용)
        self._pcd_history = {a: [] for a in self.possible_agents}
        self.dynamic_delta = 0.3            # t - δ (초) 간격
        self.dynamic_point_v_thresh = 0.5   # d/δ > 이 값이면 dynamic point
        self.labs_dyn = 5                   # dynamic point 최소 개수
        self.labs_rel = 0.3                 # dynamic point 비율 기준 (30%)
        self.enemy_v_threshold = 4.0
        self.enemy_rel_v_threshold = 5.0

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}     # {"name": np.array([x,y,z])}
        self._last_time = {}     # {"name": t_float}

        # 속도/액션 버퍼
        self.vmax_self = 2.0             # 자기 속도 상한(m/s) (학습 안정용)
        self.vmax_cluster = 15.0
        self.enemy_speed_threshold = 4.0  #  클러스터 속도 임계값 (m/s)
        self._timestep = 1.0

        # 액션 버퍼
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

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
        self.num_yaw_bins = 9
        #self.forward_speed = self.max_cmd_speed  # 전진 속도 (튜닝 가능)
        self.forward_speed = 1.5
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
        return f"{agent}_{self.lidar_name}"

    def _LidarPointsWorld(self, agent):
        """
        라이다 포인트를 '에이전트 로컬 좌표'에서 '월드 좌표'로 변환해서 리턴.
        (라이다가 기체에 고정되어 있다고 가정)
        """
        ld = self.client.getLidarData(
            lidar_name=self._lidar_sensor_name(agent),
            vehicle_name=agent
        )
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
    # 과거 포인트맵 중 t-δ 근처 참조맵 찾기
    # ======================================================================
    def _get_reference_pcd(self, agent, now_t):
        history = self._pcd_history.get(agent, [])
        if not history:
            return None, None

        target_t = now_t - self.dynamic_delta
        best = None
        best_dt = None

        for (t0, pcd0) in history:
            dt = now_t - t0
            if dt <= 1e-3:
                continue
            # t0가 target_t에 가까운 걸 고른다
            if best is None:
                best = (t0, pcd0)
                best_dt = dt
            else:
                if abs(t0 - target_t) < abs(best[0] - target_t):
                    best = (t0, pcd0)
                    best_dt = dt

        if best is None or best[1].shape[0] == 0:
            return None, None

        return best[1], best_dt  # pcd_ref, dt_ref

    # ======================================================================
    # 포인트별 dynamic 판정 (논문식 v = d/δ > thresh)
    # ======================================================================
    def _classify_dynamic_points(self, points_now, pcd_ref, dt_ref):
        """
        points_now: (N,3), 현재 클러스터 포인트 (world)
        pcd_ref:    (M,3), 과거 누적 포인트맵 (world)
        dt_ref:     now_t - t_ref (초)
        return: bool mask (N,) dynamic 여부
        """
        N = points_now.shape[0]
        if N == 0 or pcd_ref is None or pcd_ref.shape[0] == 0:
            return np.zeros(N, dtype=bool)

        try:
            nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(pcd_ref)
            dists, _ = nn.kneighbors(points_now)  # (N,1)
            dt = max(dt_ref, 1e-3)
            v_est = dists[:, 0] / dt
            dynamic_mask = v_est > self.dynamic_point_v_thresh
            return dynamic_mask
        except Exception:
            # 문제가 나면 일단 모두 static으로 처리
            return np.zeros(N, dtype=bool)

    # ======================================================================
    # 포즈/속도 관련 헬퍼
    # ======================================================================
    def _get_pose_xyz(self, name):
        pose = self.client.simGetObjectPose(vehicle_name=name).position
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
    # 라이다 데이터 → DBSCAN → 트랙/속도 추정 + dynamic/static 판정
    # ======================================================================
    def _update_clusters_and_tracks(self, agent, now_t):
        # 0) 현재 포인트맵 (world)
        pts_w = self._LidarPointsWorld(agent)

        # 히스토리에 저장 (1초 정도만 유지)
        self._pcd_history[agent].append((now_t, pts_w))
        self._pcd_history[agent] = [
            (t0, p0) for (t0, p0) in self._pcd_history[agent]
            if now_t - t0 <= 1.0
        ]

        # t-δ 근처 참조 포인트맵 가져오기 (논문식 dynamic point 판정용)
        pcd_ref, dt_ref = self._get_reference_pcd(agent, now_t)

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

                # 논문 스타일: 포인트 이동량 기반 dynamic 판정
                dyn_mask = self._classify_dynamic_points(cpts, pcd_ref, dt_ref) \
                    if (pcd_ref is not None and dt_ref is not None) else np.zeros(cpts.shape[0], dtype=bool)

                dyn_count = int(dyn_mask.sum())
                total = int(cpts.shape[0])
                is_dynamic = False
                if total > 0:
                    if dyn_count >= self.labs_dyn or dyn_count >= self.labs_rel * total:
                        is_dynamic = True

                clusters.append({
                    "centroid": centroid,
                    "points": cpts,
                    "count": total,
                    "is_dynamic": is_dynamic
                })

        # =========================
        # 2) 트랙 업데이트
        # =========================
        tracks = self._cluster_tracks[agent]

        cur_centroids = [c["centroid"] for c in clusters]
        cur_dyn_flags = [c["is_dynamic"] for c in clusters]
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
                    # 과거 centroid도 하나 넣어주는 게 안전
                    hist.append((st["last_t"], st["centroid"].copy()))
                hist.append((now_t, new_c.copy()))

                # N-stack 기반 평균 속도 계산
                if len(hist) >= 2:
                    t0, p0 = hist[0]
                    t_last, p_last = hist[-1]
                    dt_hist = max(1e-6, t_last - t0)
                    v_hist = (p_last - p0) / dt_hist
                else:
                    # 혹시 모자라면 이전 centroid와 차분
                    dt = max(1e-6, now_t - st["last_t"])
                    v_hist = (new_c - st["centroid"]) / dt

                # 클램프 + EMA
                v_hist = np.clip(v_hist, -self.vmax_cluster, self.vmax_cluster)
                prev = st.get("vel", np.zeros(3, np.float32))
                vel = 0.6 * prev + 0.4 * v_hist

                tracks[tid] = {
                    "centroid": new_c.copy(),
                    "vel": vel,
                    "last_t": now_t,
                    "is_dynamic": cur_dyn_flags[best_j],
                    "history": hist,
                    "label": st.get("label", None),
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
                    "vel": np.zeros(3, np.float32),
                    "last_t": now_t,
                    "is_dynamic": cur_dyn_flags[j],
                    "history": hist,
                    "label": None,
                }

        # =========================
        # 3) N-스택 기반 속도로 아군/적군 분류
        # =========================
        agent_pos = self._get_pose_xyz(agent)
        leader_pos = self._get_pose_xyz("Drone1")

        v_agent = self._get_self_velocity(agent, now_t, agent_pos)
        v_leader = self._get_self_velocity("Drone1", now_t, leader_pos)

        enemy_list = []
        ally_list = []

        for tid, st in tracks.items():
            # ---- N스택 평균 속도 다시 계산 (클래스 분류용) ----
            hist = st.get("history", None)
            if hist is not None and len(hist) >= 2:
                t0, p0 = hist[0]
                t_last, p_last = hist[-1]
                dt_hist = max(1e-6, t_last - t0)
                v_hist = (p_last - p0) / dt_hist
            else:
                v_hist = st.get("vel", np.zeros(3, np.float32))

            is_dynamic = bool(st.get("is_dynamic", False))

            # N-스택 기반 속도들
            speed_hist = float(np.linalg.norm(v_hist))
            rel_speed_hist_agent = float(np.linalg.norm(v_hist - v_agent))
            rel_speed_hist_leader = float(np.linalg.norm(v_hist - v_leader))

            # 판정 규칙: dynamic 이고, N-stack 속도가 빠르면 enemy
            if (
                is_dynamic and
                (
                    speed_hist > self.enemy_v_threshold or
                    rel_speed_hist_agent > self.enemy_rel_v_threshold
                )
            ):
                label = "enemy"
            else:
                label = "ally"

            st["label"] = label

            entry = {
                "id":         tid,
                "centroid":   st["centroid"].copy(),
                "vel":        v_hist.copy(),
                "last_t":     st["last_t"],
                "hist_speed": speed_hist,
                "rel_speed":  rel_speed_hist_agent,    # 디버그용
                "rel_speed_agent":  rel_speed_hist_agent,
                "rel_speed_leader": rel_speed_hist_leader,
                "is_dynamic": is_dynamic,
                "label":      label,
            }

            if label == "enemy":
                enemy_list.append(entry)
            else:
                ally_list.append(entry)

        # 에이전트에서 가까운 순으로 정렬
        def _dist(e):
            return np.linalg.norm(e["centroid"] - agent_pos)

        enemy_list.sort(key=_dist)
        ally_list.sort(key=_dist)

        self.enemy_clusters[agent] = enemy_list
        self.ally_clusters[agent] = ally_list

    # ======================================================================
    # 디버깅용: 콘솔에 적/아군 클러스터 출력
    # ======================================================================
    '''
    def _debug_print_clusters(self):
        if not self.debug_clusters:
            return

        # 너무 많이 찍히지 않게, 3스텝마다 한 번만 출력
        if self._timestep % 3 != 0:
            return

        print(f"\n[step {self._timestep}] ==== 클러스터 디버그 ====")
        for agent in self.agents:
            enemies = self.enemy_clusters.get(agent, [])
            allies = self.ally_clusters.get(agent, [])

            print(f"▶ {agent}: 적군 {len(enemies)}개, 아군 {len(allies)}개")

            for e in enemies[:5]:
                c = e["centroid"]
                v = e["vel"]
                print(f"[ENEMY] id={e['id']}, pos=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}), "
                      f"vel=({v[0]:.1f},{v[1]:.1f},{v[2]:.1f}), "
                      f"rel_speed={e['rel_speed']:.2f}, dyn={e['is_dynamic']}")

            for a in allies[:5]:
                c = a["centroid"]
                v = a["vel"]
                print(f"[ALLY ] id={a['id']}, pos=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}), "
                      f"vel=({v[0]:.1f},{v[1]:.1f},{v[2]:.1f}), "
                      f"rel_speed={a['rel_speed']:.2f}, dyn={a['is_dynamic']}")
    '''
    # ======================================================================
    # 속도 추정
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
            self.client.enableApiControl(True, vehicle_name="Drone1")
            self.client.armDisarm(True, vehicle_name="Drone1")
            for agent in self.possible_agents:
                self.client.enableApiControl(True, vehicle_name=agent)
                self.client.armDisarm(True, vehicle_name=agent)

            # 동적장애물 제어 추가
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
        유인기를 목적지로 이동시키고, 도착 시 성공 여부 반환.
        moveByVelocityZAsync를 사용해서 실제 속도로 계속 움직이게 함.
        Returns:
            mission_accomplished (bool)
        """
        if not self.leader_waypoints:
            self._generate_leader_waypoints()

        target = self.leader_waypoints[0]

        pose = self.client.simGetVehiclePose(vehicle_name="Drone1")
        cur = np.array([pose.position.x_val,
                        pose.position.y_val,
                        pose.position.z_val], dtype=np.float32)

        dist_to_target = np.linalg.norm(target[:2] - cur[:2])

        if dist_to_target < 3.0:
            self.client.moveByVelocityZAsync(
                0.0, 0.0,
                2.0,
                duration=self.dt,
                vehicle_name="Drone1"
            )
            return True  # 미션 성공

        dir_vec = target - cur
        dist = np.linalg.norm(dir_vec[:2])
        if dist > 1e-6:
            dir_unit = dir_vec / (dist + 1e-9)

            vx = float(dir_unit[0] * 0.5)
            vy = float(dir_unit[1] * 0.5)
            vz = float(self.fixed_z)

            self.client.moveByVelocityZAsync(
                vx, vy,
                vz,
                duration=self.dt,
                vehicle_name="Drone1"
            )

        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.1:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

        return False

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
    def _update_dynamic_obstacle(self, t):
        name = self.dynamic_name
        fixed_z = self.fixed_z
        attack_speed = 5.0
        STOP_DISTANCE = 1.0

        if not hasattr(self, "_obstacle_state"):
            self._obstacle_state = "IDLE"
            self._next_chase_time = time.time() + random.uniform(0.0, 1.0)
            self._idle_pos = None
            self._chase_mode = None

        leader_pose = self.client.simGetObjectPose("Drone1").position
        obstacle_pose = self.client.simGetObjectPose(name).position
        lx, ly, lz = leader_pose.x_val, leader_pose.y_val, leader_pose.z_val
        cx, cy, cz = obstacle_pose.x_val, obstacle_pose.y_val, obstacle_pose.z_val

        dx, dy = lx - cx, ly - cy
        dist_2d = math.sqrt(dx ** 2 + dy ** 2) + 1e-9

        if self._obstacle_state == "IDLE":
            if self._idle_pos is None:
                radius = random.uniform(90.0, 100.0)
                angle = random.uniform(0, 2 * np.pi)
                self._idle_pos = (lx + radius * np.cos(angle), ly + radius * np.sin(angle))
                print(f"[대기모드] {radius:.1f}m 거리에서 대기")

            ix, iy = self._idle_pos
            dx_i, dy_i = ix - cx, iy - cy
            dist_idle = math.sqrt(dx_i ** 2 + dy_i ** 2)
            if dist_idle > 1.0:
                vx = dx_i / dist_idle * 2.0
                vy = dy_i / dist_idle * 2.0
                vz = (fixed_z - cz) * 0.3
                self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1, vehicle_name=name)
            else:
                self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=name)

            if time.time() > self._next_chase_time:
                self._obstacle_state = "CHASE"
                self._chase_start = time.time()
                self._chase_mode = random.choice(["STRAIGHT"])
                print(f"[추격 시작] 모드: {self._chase_mode}")
                return

        elif self._obstacle_state == "CHASE":
            elapsed = time.time() - self._chase_start
            chase_duration = random.uniform(10.0, 11.0)

            if elapsed > chase_duration:
                print(f"[추격 종료] ({self._chase_mode}) {elapsed:.1f}s 후 대기 복귀")
                self._obstacle_state = "RETURN"
                self._idle_pos = None
                self._next_chase_time = time.time() + random.uniform(0.0, 0.0)
                return

            if dist_2d <= STOP_DISTANCE:
                self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=name)
                self.leader_stop = True
                return

            dir_x = dx / dist_2d
            dir_y = dy / dist_2d

            if self._chase_mode == "STRAIGHT":
                vx = dir_x * attack_speed
                vy = dir_y * attack_speed
            else:
                amplitude = 10.0
                freq = 1.0
                phase = math.sin(2.0 * math.pi * freq * t / 2)
                perp_x = -dir_y
                perp_y = dir_x
                vx = (dir_x * attack_speed) + (perp_x * amplitude * phase)
                vy = (dir_y * attack_speed) + (perp_y * amplitude * phase)

            vz = (fixed_z - cz) * 8.0
            self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1, vehicle_name=name)

        elif self._obstacle_state == "RETURN":
            if self._idle_pos is None:
                radius = random.uniform(90.0, 100.0)
                angle = random.uniform(0, 2 * np.pi)
                self._idle_pos = (lx + radius * np.cos(angle), ly + radius * np.sin(angle))

            ix, iy = self._idle_pos
            dx_r, dy_r = ix - cx, iy - cy
            dist_return = math.sqrt(dx_r ** 2 + dy_r ** 2)
            if dist_return > 1.0:
                vx = dx_r / dist_return * 4.0
                vy = dy_r / dist_return * 4.0
                vz = (fixed_z - cz) * 8.0
                self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1, vehicle_name=name)
            else:
                self._obstacle_state = "IDLE"
                self._next_chase_time = time.time() + random.uniform(1.0, 3.0)
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

    # ======================================================================
    # PettingZoo API-ish
    # ======================================================================
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
        self._cluster_tracks = {a: {} for a in self.possible_agents}
        self._next_cluster_id = {a: 0 for a in self.possible_agents}
        self.enemy_clusters = {a: [] for a in self.possible_agents}
        self.ally_clusters = {a: [] for a in self.possible_agents}
        self._pcd_history = {a: [] for a in self.possible_agents}

        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1")
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent)

        now_t = 0.0
        obs_list = [self._get_obs(a, now_t, all_poses) for a in self.agents]

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

        return obs

    def _do_action(self, agent, action):
        """
        이산 행동:
          action: int in [0, num_yaw_bins-1]
            → yaw_norm ∈ [-1, 1] 로 매핑
            → yaw_rate = yaw_norm * max_yaw_rate
            → 고정 속도(self.forward_speed)로 yaw 방향으로 전진
        """
        if isinstance(action, (np.ndarray, list, tuple)):
            a_idx = int(action[0])
        else:
            a_idx = int(action)
        a_idx = np.clip(a_idx, 0, self.num_yaw_bins - 1)

        if self.num_yaw_bins > 1:
            ratio = a_idx / (self.num_yaw_bins - 1)
        else:
            ratio = 0.5
        yaw_norm = -1.0 + 2.0 * ratio  # [-1, 1]

        self._last_action[agent] = np.array([yaw_norm, 0.0], dtype=np.float32)
        self._current_action[agent] = self._last_action[agent].copy()

        max_yaw_rate_deg = 90.0  # deg/s
        yaw_rate = math.radians(max_yaw_rate_deg) * yaw_norm  # [rad/s]

        dt_turn = self.dt * 0.5
        dt_forward = self.dt * 0.5

        self.client.moveByRollPitchYawrateZAsync(
            roll=0.0,
            pitch=0.0,
            yaw_rate=yaw_rate,
            z=float(self.fixed_z),
            duration=dt_turn,
            vehicle_name=agent
        ).join()

        pose = self.client.simGetVehiclePose(vehicle_name=agent)
        _, _, yaw = airsim.to_eularian_angles(pose.orientation)

        speed = self.forward_speed  # [m/s]

        vx = speed * math.cos(yaw)
        vy = speed * math.sin(yaw)

        self.client.moveByVelocityZAsync(
            vx=vx,
            vy=vy,
            z=float(self.fixed_z),
            duration=dt_forward,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw)),
            vehicle_name=agent
        )
    

    def step(self, actions):
        self._timestep += 1
        t = time.time()
        self._update_dynamic_obstacle(t)
        now_t = self._timestep * self.dt
        # ================================
        # 🔥 (0) AirSim 충돌 이벤트 검사 (가장 먼저 처리)
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

        # 디버그 보고 싶으면 주석 해제
        # self._debug_print_clusters()

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
        for i in range(n):
            info_i = per_agent_infos[i].copy()
            info_i.update(final_team_info)
            infos_list.append(info_i)

        return obs_list, rewards_list, dones_list, infos_list
