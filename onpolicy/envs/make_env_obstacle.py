# from MARL_test import ParallelEnv
import gym
import numpy as np
import airsim
import math
import time
import random
from sklearn.cluster import DBSCAN

# import open3d as o3d
# from sklearn.datasets import

class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",                 # 기본값
        follower_names=("Follower0","Follower1","Follower2"),
        lidar_name="LidarSensor",               # 라이다 공통명 (실제 센서는 f"{agent}_{lidar_name}")
        min_samples=5,
        step_length=1.0,
        fixed_z=-10.0,
        leader_velocity=0.0,                    # 유인기 속도(m/s) 파라미터화
        optimal_distance=10.0,                  # 추종 최적 거리(시각화용 링)
        far_cutoff=60.0,                        # 이탈 종료 거리(시각화용 링)
        too_close=0.5,                          # 유인기와 최소 거리
        dt=0.05,                                # 내부 시뮬레이션 타임스텝(초)
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
        self.K_nearest = 4                  # K-최근접 동적 객체 개수
        self.match_dist_max = 2.0           # 프레임 간 클러스터 매칭 허용 거리[m]
        self._cluster_tracks = {a: {} for a in self.possible_agents}  # agent별 트랙 사전
        self._next_cluster_id = {a: 0 for a in self.possible_agents}  # agent별 ID 증가기

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}     # {"name": np.array([x,y,z])}
        self._last_time = {}     # {"name": t_float}

        # 속도/액션 버퍼
        self.vmax_self = 10.0             # 자기 속도 상한(m/s) (학습 안정용)
        self.vmax_cluster = 15.0
        self._timestep = 0

        # KeyError 방지: 액션 버퍼를 에이전트별로 초기화
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        # 이동 모드
        self.use_teleport = True  # True면 위치 보정(텔레포트) 기반 이동

        # 클라이언트
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = float(fixed_z)
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)

        self.leader_velocity = float(leader_velocity)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)

        self._first_setup = True
        self.leader_stop = False

        # ===== obs / act / share_obs spaces =====
        # obs_dim = [self_vel(3) + rel_leader(3) + K * (rel_pos(3)+rel_vel(3))]
        obs_dim = 3 + 3 + self.K_nearest * (3 + 3)   # = 30
        act_dim = 2
        share_obs_dim = obs_dim * len(self.possible_agents)

        low_self = [-20.0, -20.0, -20.0]             # 속도/상대위치 대략적 클립
        high_self = [ 20.0,  20.0,  20.0]
        low_rel = [-200.0, -200.0, -200.0]
        high_rel = [ 200.0,  200.0,  200.0]
        low_kn = [-200.0]*3 + [-20.0]*3
        high_kn = [ 200.0]*3 + [ 20.0]*3
        per_agent_low  = low_self + low_rel + (low_kn * self.K_nearest)
        per_agent_high = high_self + high_rel + (high_kn * self.K_nearest)

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low,  dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,), dtype=np.float32
            ) for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
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
        self._last_visualize_t = time.time()

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

        # 기존 트랙 업데이트/삭제
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
                vel = 0.6 * prev + 0.4 * vel
                tracks[tid] = {"centroid": new_c, "vel": vel, "last_t": now_t}
                cur_used[best_j] = True
            else:
                if now_t - st["last_t"] > 1.0:
                    del tracks[tid]

        # 매칭 안 된 클러스터는 신규 트랙 생성
        for j, cen in enumerate(cur_centroids):
            if not cur_used[j]:
                tid = self._next_cluster_id[agent]
                self._next_cluster_id[agent] += 1
                tracks[tid] = {"centroid": cen, "vel": np.zeros(3, np.float32), "last_t": now_t}

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
            # 텔레포트 모드 (기존과 동일, API 호출 없음)
            a = self._last_action[name] 
            vx = float(a[0]) * float(self.step_length) / self.dt
            vy = float(a[1]) * float(self.step_length) / self.dt
            vz = 0.0
            v = np.array([vx, vy, vz], dtype=np.float32)
        else:
            # (수정) API 호출 대신 인자로 받은 'current_pos' 사용
            pos = current_pos 
            v = np.zeros(3, np.float32)
            if name in self._last_pose:
                dt = max(1e-6, now_t - self._last_time.get(name, now_t))
                v = (pos - self._last_pose[name]) / dt
            self._last_pose[name] = pos
            self._last_time[name] = now_t

        # 학습 안정화를 위한 클램프 & EMA (기존과 동일)
        v = np.clip(v, -self.vmax_self, self.vmax_self)
        prev = getattr(self, "_vel_ema_" + name, v)
        v_ema = 0.7 * prev + 0.3 * v
        setattr(self, "_vel_ema_" + name, v_ema)
        return v_ema

    def _get_knn_features(self, agent, K, now_t, all_poses):
        """
        _get_pose_xyz() API 호출을 제거하고,
        미리 계산된 all_poses 딕셔너리를 사용
        """
        
        # (수정) API 호출 대신 딕셔너리 조회
        pos_a = all_poses[agent]
        
        # (수정) 최적화된 속도 함수 호출
        vel_a = self._get_self_velocity(agent, now_t, pos_a)

        feats = []
        cand = []
        # (이하 로직은 기존과 동일)
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

    # ======================================================================
    # Spaces (PettingZoo-style getters)
    # ======================================================================
    # def _get_lidar_obs(self, agent, lidar_name="LidarSensor1"):
    #     lidar_data = self.client.getLidarData(vehicle_name=agent, lidar_name=lidar_name)
    #     if len(lidar_data.point_cloud) < 3:
    #         return np.full(36, self.far_cutoff, dtype=np.float32)

    #     pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    #     dists = np.linalg.norm(pts[:, :2], axis=1)
    #     angles = np.arctan2(pts[:, 1], pts[:, 0])

    #     bins = np.linspace(-math.pi, math.pi, 37)
    #     min_dists = np.full(36, self.far_cutoff, dtype=np.float32)

    #     for i in range(36):
    #         sel = dists[(angles >= bins[i]) & (angles < bins[i+1])]
    #         if len(sel) > 0:
    #             min_dists[i] = np.min(sel)
    #     return min_dists
    
    def _get_lidar_obs(self, agent):
        lidar_data = self.client.getLidarData(
            vehicle_name=agent,
            lidar_name=self._lidar_sensor_name(agent)
        )
        if len(lidar_data.point_cloud) < 3:
            return np.full(36, self.far_cutoff, dtype=np.float32)

        pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        dists = np.linalg.norm(pts[:, :2], axis=1)
        angles = np.arctan2(pts[:, 1], pts[:, 0])

        bins = np.linspace(-math.pi, math.pi, 37)
        min_dists = np.full(36, self.far_cutoff, dtype=np.float32)

        for i in range(36):
            sel = dists[(angles >= bins[i]) & (angles < bins[i+1])]
            if len(sel) > 0:
                min_dists[i] = np.min(sel)
        return min_dists


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

            # ✅ 동적장애물 제어 추가
            self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
            self.client.armDisarm(True, vehicle_name=self.dynamic_name)

            futs = [self.client.takeoffAsync(vehicle_name="Drone1")]
            futs += [self.client.takeoffAsync(vehicle_name=a) for a in self.possible_agents]
            futs += [self.client.takeoffAsync(vehicle_name=self.dynamic_name)]
            for f in futs: f.join()

            time.sleep(1.0)
            self._teleport_to_start()
            self._first_setup = False

    def _teleport_to_start(self):
        leader_start_x, leader_start_y = 5.0, 2.5
        self.leader_home_pose = airsim.Pose(
            airsim.Vector3r(float(leader_start_x), float(leader_start_y), float(self.fixed_z)),
            airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)
        )
        radius = random.uniform(20.0, 40.0) 
        angle = random.uniform(0, 2 * np.pi)
        obstacle_start_x = leader_start_x + radius * np.cos(angle)
        obstacle_start_y = leader_start_y + radius * np.sin(angle)
        # settings.json에 맞춘 시작 좌표 (X, Y, Z)
        start_cfg = {
            "Drone1":   (5.0,  2.5, self.fixed_z),
            "Follower0":(0.0,  0.0, self.fixed_z),
            "Follower1":(0.0,  2.5, self.fixed_z),
            "Follower2":(0.0,  5.0, self.fixed_z),
            # 동적장애물 시작 위치
            self.dynamic_name: (obstacle_start_x, obstacle_start_y, self.fixed_z),
        }
        # API 제어 보장
        self.client.enableApiControl(True, vehicle_name="Drone1")
        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
        # 텔레포트
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


    def _update_leader_movement(self):
        """
        유인기를 현재 X,Y 위치의 self.fixed_z에 "고정" (추락 방지).
        시각화 담당.
        """
        # --- [추가] 추락 방지를 위한 위치 고정 ---
        try:
            # (수정) 현재 위치(Get)를 읽지 않고, 저장된 '홈' 위치(Set)를 강제 적용
            self.client.simSetVehiclePose(
                self.leader_home_pose, # (5.0, 2.5, self.fixed_z)
                ignore_collision=True,
                vehicle_name="Drone1"
            )
        except Exception as e:
            # 시뮬레이션 연결이 끊겼을 때를 대비한 예외 처리
            print(f"Leader(Drone1) 위치 고정 중 오류: {e}")
        # --- [추가 끝] ---


        # 시각화 (유인기 위치는 _visualize_circles 내부에서 직접 조회)
        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.1:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

        # 미션 성공(True)을 반환하지 않음
        return False

    # --------------------- 시각화 ---------------------
    def _visualize_circles(self):
        try:
            leader_pos = self.client.simGetObjectPose("Drone1").position
            center = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val], dtype=float)

            def ring_points(radius, n=36):
                pts = []
                for i in range(n+1):
                    ang = (i / n) * 2 * np.pi
                    x = center[0] + radius * np.cos(ang)
                    y = center[1] + radius * np.sin(ang)
                    z = center[2]
                    pts.append(airsim.Vector3r(x, y, z))
                return pts

            line_thickness = 20.0
            self.client.simPlotLineStrip(ring_points(self.optimal_distance), [1, 1, 0, 0.8], line_thickness, 0.15, True)
            self.client.simPlotLineStrip(ring_points(self.far_cutoff), [0, 1, 0, 0.8], line_thickness, 0.15, True)
        except Exception:
            pass


    # ======================================================================
    # 관측/액션/보상
    # ======================================================================
    def _get_obs(self, agent, now_t, all_poses):
        now_t = self._timestep * self.dt
        """
        API 호출 없이 all_poses 딕셔너리를 사용해 관측값을 생성
        """
    
        # 1) 자신 속도
        # (수정) _get_pose_xyz를 호출하지 않도록, 현재 위치(all_poses[agent])를 인자로 전달
        self_vel = self._get_self_velocity(agent, now_t, all_poses[agent])

        # 2) 유인기 상대 위치 (리더-에이전트)
        # (수정) API 호출 대신 딕셔너리 조회
        leader_pos = all_poses["Drone1"]
        agent_pos  = all_poses[agent]
        rel_la = (leader_pos - agent_pos).astype(np.float32)

        # 3) K-최근접 동적 객체: 상대 위치 + 상대 속도
        # (수정) _get_knn_features가 딕셔너리를 사용하도록 수정
        kn_feats = self._get_knn_features(agent, self.K_nearest, now_t, all_poses)

        obs = np.concatenate([self_vel, rel_la, kn_feats], axis=0).astype(np.float32)
        return obs

    def _do_action(self, agent, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action[agent] = a.copy()
        self._current_action[agent] = a.copy()  # ★ 현재 액션 기록 (KeyError 방지)

        pose = self.client.simGetVehiclePose(agent)
        x, y, z = pose.position.x_val, pose.position.y_val, self.fixed_z

        dx = float(a[0]) * float(self.step_length)
        dy = float(a[1]) * float(self.step_length)

        nx = float(x) + dx
        ny = float(y) + dy
        nz = float(z)

        new_pos = airsim.Vector3r(nx, ny, nz)
        new_pose = airsim.Pose(new_pos, airsim.Quaternionr(0.0, 0.0, 0.0, 1.0))
        self.client.simSetVehiclePose(new_pose, False, vehicle_name=agent)
  
    # --------------------- 보상 ---------------------
    def _compute_reward(self, agent, all_poses):
        agent_pos  = all_poses[agent]
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
            # 최적 거리 구간: 최고 보상
            reward = 3.0
        elif 0.5 <= dist < 5.0:
            # 0.5~5m: 가까워질수록 급감 (-150 → +3)
            reward = -150.0 + (153.0) * math.exp(-((dist - 5.0)**2) / (2 * 1.5**2))
        elif 10.0 < dist <= 60.0:
            # 10~60m: 멀어질수록 감점 (+3 → -150)
            reward = -150.0 + (153.0) * math.exp(-((dist - 10.0)**2) / (2 * 15.0**2))
        else:
            reward = -150.0

        if self.leader_stop:
            return -150.0, True

        return float(reward), False

    # --------------------- 팀 보상 집계 ---------------------
    # 팀 보상 집계(평균) + 안전 게이트
    def _team_reward_and_done(self, per_agent_results, mission_accomplished=False):
        # 안전 게이트: 한 명이라도 종료이면 즉시 실패
        any_fail = any(done_i for (_, done_i) in per_agent_results)
        if any_fail:
            return -200.0, True, {"final_status": "FAIL_CRASH"}

        # 미션 성공: 유인기가 피격당하지 않는 것.
        if mission_accomplished:
            tracking_rewards = [r_i for (r_i, _) in per_agent_results]
            final_reward = float(np.mean(tracking_rewards)) + 500.0
            return final_reward, True, {"final_status": "SUCCESS"}

        # 미션 진행 중: 추종 보상의 평균
        tracking_rewards = [r_i for (r_i, _) in per_agent_results]
        return float(np.mean(tracking_rewards)), False, {}
    

    # --------------------- 동적장애물 FSM + 직선/곡선 추격 ---------------------
    def _update_dynamic_obstacle(self, t):
        name = self.dynamic_name
        fixed_z = self.fixed_z
        attack_speed = 5.0
        STOP_DISTANCE = 1.0

        # 상태 변수 초기화
        if not hasattr(self, "_obstacle_state"):
            self._obstacle_state = "IDLE"
            self._next_chase_time = time.time() + random.uniform(1.0, 3.0)
            self._idle_pos = None
            self._chase_mode = None  # 'STRAIGHT' or 'CURVED'

        leader_pose = self.client.simGetObjectPose("Drone1").position
        obstacle_pose = self.client.simGetObjectPose(name).position
        lx, ly, lz = leader_pose.x_val, leader_pose.y_val, leader_pose.z_val
        cx, cy, cz = obstacle_pose.x_val, obstacle_pose.y_val, obstacle_pose.z_val

        dx, dy = lx - cx, ly - cy
        dist_2d = math.sqrt(dx**2 + dy**2) + 1e-9

        # ------------------ IDLE 모드 ------------------
        if self._obstacle_state == "IDLE":
            if self._idle_pos is None:
                radius = random.uniform(60.0, 70.0)
                angle = random.uniform(0, 2 * np.pi)
                self._idle_pos = (lx + radius * np.cos(angle), ly + radius * np.sin(angle))
                print(f"[대기모드] {radius:.1f}m 거리에서 대기")

            ix, iy = self._idle_pos
            dx_i, dy_i = ix - cx, iy - cy
            dist_idle = math.sqrt(dx_i**2 + dy_i**2)
            if dist_idle > 1.0:
                vx = dx_i / dist_idle * 2.0
                vy = dy_i / dist_idle * 2.0
                vz = (fixed_z - cz) * 0.3
                self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1, vehicle_name=name)
            else:
                self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=name)

            # 일정 시간 후 추격 시작
            if time.time() > self._next_chase_time:
                self._obstacle_state = "CHASE"
                self._chase_start = time.time()
                self._chase_mode = random.choice(["STRAIGHT", "CURVED"])
                print(f"[추격 시작] 모드: {self._chase_mode}")
                return

        # ------------------ CHASE 모드 ------------------
        elif self._obstacle_state == "CHASE":
            elapsed = time.time() - self._chase_start
            chase_duration = random.uniform(15.0, 16.0)

            if elapsed > chase_duration:
                print(f"[추격 종료] ({self._chase_mode}) {elapsed:.1f}s 후 대기 복귀")
                self._obstacle_state = "RETURN"
                self._idle_pos = None
                self._next_chase_time = time.time() + random.uniform(1.0, 3.0)
                return

            if dist_2d <= STOP_DISTANCE:
                # print(f"[정지] Drone1과 거의 일치 (거리={dist_2d:.2f}m)")
                vz = (fixed_z - cz )*8.0
                self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=name)
                self.leader_stop = True
                return

            # === [수정된 핵심 로직] ===
            # 'CURVED' 모드일 때 5m 이내로 접근하면, 'STRAIGHT' 모드로 "상태를 변경"
            if self._chase_mode == "CURVED" and dist_2d < 5.0:
                print(f"[곡선->직선] 2m 이내({dist_2d:.1f}m), 직선 돌격 모드로 변경!")
                self._chase_mode = "STRAIGHT"
            # === [수정 끝] ===


            dir_x = dx / dist_2d
            dir_y = dy / dist_2d
            
            # --- (1) 직선 추격 ---
            # (이제 self._chase_mode가 'STRAIGHT'로 바뀌었기 때문에 이 로직이 실행됨)
            if self._chase_mode == "STRAIGHT":
                vx = dir_x * attack_speed
                vy = dir_y * attack_speed

            # --- (2) 곡선 추격 ---
            elif self._chase_mode == "CURVED": # (self._chase_mode == "CURVED" 이고 아직 2m 밖일 때)
                amplitude = 10.0
                freq = 1.0
                phase = math.sin(2.0 * math.pi * freq * t/2)
                perp_x = -dir_y
                perp_y = dir_x
                vx = (dir_x * attack_speed) + (perp_x * amplitude * phase)
                vy = (dir_y * attack_speed) + (perp_y * amplitude * phase)

            vz = (fixed_z - cz )*8.0
            self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1, vehicle_name=name)


        # ------------------ RETURN 모드 ------------------
        elif self._obstacle_state == "RETURN":
            if self._idle_pos is None:
                radius = random.uniform(30.0, 50.0)
                angle = random.uniform(0, 2 * np.pi)
                self._idle_pos = (lx + radius * np.cos(angle), ly + radius * np.sin(angle))
                # print(f"[대기 위치 재설정] 새 지점으로 복귀 중")

            ix, iy = self._idle_pos
            dx_r, dy_r = ix - cx, iy - cy
            dist_return = math.sqrt(dx_r**2 + dy_r**2)
            if dist_return > 1.0:
                vx = dx_r / dist_return * 3.0
                vy = dy_r / dist_return * 3.0
                vz = (fixed_z - cz) * 8.0
                self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1, vehicle_name=name)
            else:
                self._obstacle_state = "IDLE"
                self._next_chase_time = time.time() + random.uniform(1.0, 3.0)
                # print("[대기모드 복귀 완료]")




    # ======================================================================
    # PettingZoo API-ish
    # ======================================================================   
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._setup_flight()
        self.current_waypoint_idx = 0
        self._teleport_to_start() # <- 드론 위치가 여기서 설정됨
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

        # 'step' 함수에서처럼 'now_t'와 'all_poses'를 생성
        # now_t = self._timestep * self.dt  # (now_t = 0.0)
        
        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1") # 유인기
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent) # 에이전트

        now_t = 0.0
        obs_list = [self._get_obs(a, now_t, all_poses) for a in self.agents]

        self._obstacle_state = "IDLE"
        self._idle_pos = None
        self._next_chase_time = time.time() + random.uniform(1.0, 3.0)
        self._chase_mode = None
        self.leader_stop = False
        # print("[Reset] 동적장애물 상태 초기화 완료")
        return obs_list
    

    def step(self, actions):
        self._timestep += 1
        t = time.time()  # 현재 시간 저장
        self._update_dynamic_obstacle(t)
        now_t = self._timestep * self.dt

        # A) LiDAR 기반 트랙 갱신
        for agent in self.agents:
            if agent.startswith("Follower"):
                self._update_clusters_and_tracks(agent, now_t)

        # B) 액션 적용
        for agent, act in zip(self.agents, actions):
            self._do_action(agent, act)

        # C) 리더 이동/시각화
        self._update_leader_movement()
        mission_accomplished = False

        # 모든 위치 정보를 이 시점에 "한 번만" 가져옴
        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1") # 유인기
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent) # 에이전트

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

        # ✅ 가디언(보호) 로직
        leader_pos   = self._get_pose_xyz("Drone1")
        obstacle_pos = self._get_pose_xyz(self.dynamic_name)


        if self.leader_stop:
            print("💥[리더 피격 감지] leader_stop=True → 에피소드 종료")

            # 관측 크기를 미리 가져오기
            obs_dim = self.observation_spaces[self.agents[0]].shape[0]
            zero_obs = np.zeros(obs_dim, dtype=np.float32)

            obs_list = [zero_obs.copy() for _ in self.agents]
            rewards_list = [-150.0] * len(self.agents)
            dones_list = [True] * len(self.agents)
            infos_list = [{"event": "leader_stop_triggered"} for _ in self.agents]
            return obs_list, rewards_list, dones_list, infos_list

        
        # 팔로워(가디언)가 막은 경우
        for agent in self.agents:
            agent_pos = self._get_pose_xyz(agent)
            if np.linalg.norm(agent_pos[:2] - obstacle_pos[:2]) < 0.5:
                print(f"🛡️[가디언 차폐 성공] {agent} ↔ DynamicObstacle (+3)")
                # 성공 보너스 +500점 부여
                rewards_list = [500.0] * len(self.agents) 
                # 즉시 에피소드 종료
                dones_list = [True] * len(self.agents)  
                infos_list = [{"event": "mission_success_guardian_block", "blocker": agent} for _ in self.agents]
                return obs_list, rewards_list, dones_list, infos_list
        
        return obs_list, rewards_list, dones_list, infos_list
    
