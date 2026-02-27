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
        leader_step_length=0.3,                    # 유인기 속도(m/s) 파라미터화
        optimal_distance=10.0,                  # 추종 최적 거리(시각화용 링)
        far_cutoff=60.0,                        # 이탈 종료 거리(시각화용 링)
        too_close=1.0,                          # 유인기와 최소 거리
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

        # ★ KeyError 방지: 액션 버퍼를 에이전트별로 초기화
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

        self.leader_step_length = float(leader_step_length)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)

        self._first_setup = True

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
        self._setup_flight()
        self._generate_leader_waypoints()
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
        [최적화됨] _get_pose_xyz() API 호출을 제거하고,
        미리 계산된 all_poses 딕셔너리를 사용합니다.
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
            # 리더 + 팔로워 API 제어 / 암
            self.client.enableApiControl(True, vehicle_name="Drone1")
            self.client.armDisarm(True, vehicle_name="Drone1")
            for agent in self.possible_agents:
                self.client.enableApiControl(True, vehicle_name=agent)
                self.client.armDisarm(True, vehicle_name=agent)

            futs = [self.client.takeoffAsync(vehicle_name="Drone1")]
            futs += [self.client.takeoffAsync(vehicle_name=a) for a in self.possible_agents]
            for f in futs:
                f.join()
            time.sleep(1.0)
            self._teleport_to_start()
            self._first_setup = False

    def _teleport_to_start(self):
        # settings.json에 맞춘 시작 좌표 (X, Y, Z)
        start_cfg = {
            "Drone1":   (5.0,  2.5, float(self.fixed_z)),
            "Follower0":(0.0,  0.0, float(self.fixed_z)),
            "Follower1":(0.0,  2.5, float(self.fixed_z)),
            "Follower2":(0.0,  5.0, float(self.fixed_z)),
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

    # ======================================================================
    # 유인기 경로/이동/시각화
    # ======================================================================
    def _generate_leader_waypoints(self):
        leader_start_pos = np.array([5.0, 2.5, self.fixed_z])

        # 시작점으로부터 50~70m 떨어진 곳에 무작위 목적지 설정
        distance = random.uniform(50.0, 70.0)
        angle = random.uniform(0, 2 * np.pi)

        final_destination = leader_start_pos + np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            0.0
        ])

        self.leader_waypoints = [final_destination]
        self.current_waypoint_idx = 0

        # 목적지 시각화용 오브젝트 이동(있을 때만)
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
        Returns:
            mission_accomplished (bool)
        """
        if not self.leader_waypoints:
            self._generate_leader_waypoints()

        target = self.leader_waypoints[0]

        pose = self.client.simGetVehiclePose(vehicle_name="Drone1")
        cur = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])

        # 목적지 도착 판정 (XY 거리 기준)
        dist_to_target = np.linalg.norm(target[:2] - cur[:2])
        if dist_to_target < 3.0:
            return True  # 미션 성공!

        # 목적지를 향해 이동
        dir_vec = target - cur
        dist = np.linalg.norm(dir_vec[:2])
        if dist > 1e-6:
            dir_unit = dir_vec / (dist + 1e-9)
            move = dir_unit * self.leader_step_length / 2
            new_pos = cur + move
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(new_pos[0], new_pos[1], self.fixed_z),
                    airsim.Quaternionr()
                ),
                ignore_collision=True,
                vehicle_name="Drone1"
            )
        

        # 시각화
        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.5:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

        return False


    def _visualize_circles(self):
        try:
            # 1. 리더 위치 가져오기: simGetObjectPose 사용으로 수정
            # 'Drone1'이 시뮬레이션 환경의 객체 이름이라고 가정
            leader_pose = self.client.simGetObjectPose(object_name="Drone1")
            if leader_pose is None:
                # 객체 포즈를 가져올 수 없는 경우 시각화 건너뛰기
                return 
            
            leader_pos = leader_pose.position
            
            # 중심 좌표 (Numpy 배열로 변환)
            center = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val], dtype=float)

            def ring_points(radius, n=36):
                """주어진 반지름과 중심을 사용하여 링의 점들을 계산합니다."""
                pts = []
                for i in range(n + 1):
                    ang = (i / n) * 2 * np.pi
                    x = center[0] + radius * np.cos(ang)
                    y = center[1] + radius * np.sin(ang)
                    z = center[2] # 리더의 Z 위치를 그대로 사용
                    pts.append(airsim.Vector3r(x, y, z))
                return pts

            # 시각화 설정
            line_thickness = 15.0
            duration = 0.1 # 그리는 지속 시간
            color = [1, 1, 0, 0.8] 

            # 2. 5m 링 그리기 (동일 색상)
            self.client.simPlotLineStrip(ring_points(5.0), color, line_thickness, duration, True)
            
            # 3. 10m 링 그리기 (동일 색상)
            self.client.simPlotLineStrip(ring_points(10.0), color, line_thickness, duration, True)
            
        except Exception as e:
            # 시각화 실패 무시 (디버깅을 위해 예외를 출력할 수도 있음)
            # print(f"Visualization failed: {e}")
            pass

    # ======================================================================
    # 관측/액션/보상
    # ======================================================================
    def _get_obs(self, agent, now_t, all_poses):
        now_t = self._timestep * self.dt
        """
        [최적화됨] API 호출 없이 all_poses 딕셔너리를 사용해 관측값을 생성합니다.
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
  

    def _compute_reward(self, agent, all_poses):
        # === 거리 및 위치 계산 ===
        leader_pos = all_poses["Drone1"]
        agent_pos = all_poses[agent]
        
        # 리더와의 거리 (3D 벡터 노름 사용 가정)
        dist = np.linalg.norm(leader_pos - agent_pos) 
        #-------------------------[수정한 부분]-------------------------------
        repulsion_penalty = 0.0
        MIN_SAFE_DISTANCE = 3.0  # 안전 거리 (예: 3m)
        
        for other in self.agents:
            if other == agent:
                continue
            
            other_pos = all_poses[other]
            dist_agents = np.linalg.norm([agent_pos[0] - other_pos[0],
                                        agent_pos[1] - other_pos[1]])
            
            # (1) 하드 충돌 (기존 로직 유지)
            if dist_agents < 0.5:
                print(f"💥 {agent}와 {other} 충돌!")
                return -150.0, True  # 즉시 종료

            # (2) 안전 거리 침범 시 연속적인 페널티
            if dist_agents < MIN_SAFE_DISTANCE:
                # 거리가 0.5m에 가까워질수록 페널티가 기하급수적으로 증가
                # 3m일 때: -0.1 * (3.0 - 0.5) / (3.0 - 0.5) = -0.1
                # 0.5m 직전일 때: 매우 큰 음수
                repulsion_penalty -= 0.1 / (dist_agents - 0.49) # 0.5m에서 발산하도록 (0.49는 하이퍼파라미터)
        #-------------------------[여기까지 수정]-------------------------------
        
        # === (2) 리더와의 거리 이탈/충돌 처리 ===
        if dist < 0.5 or dist > 60.0:
            print("💤리더와의 거리 너무 멀거나 가까움!")
            return -150.0, True  # 리더와의 거리 이탈/충돌 시 즉시 종료

        # === (3) 거리 보상 계산 (비종료 구간: 0.5m < dist <= 60.0m) ===
        
        if 5.0 <= dist <= 10.0:
            # 최적 거리 구간: 최고 보상 (고정 +3.0)
            dist_reward = 3.0
        elif 0.5 < dist < 5.0:
            # 0.5m~5m: 5m를 중심으로, 0.5m에서 -150, 5m에서 +3으로 수렴하는 가우시안
            dist_reward = -150.0 + (153.0) * math.exp(-((dist - 5.0)**2) / (2 * 1.5**2))
            dist_reward = np.clip(dist_reward, -150.0, 3.0)
        elif 10.0 < dist <= 60.0:
            # 10m~60m: 10m를 중심으로, 60m에서 -150, 10m에서 +3으로 수렴하는 가우시안
            dist_reward = -150.0 + (153.0) * math.exp(-((dist - 10.0)**2) / (2 * 15.0**2))
            dist_reward = np.clip(dist_reward, -150.0, 3.0)
        else:
            # 이 else 문은 0.5m < dist <= 60.0m 조건 내에서 불가능함 (논리적 보수성)
            dist_reward = -150.0 

        # === (4) 부드러운 이동 페널티 (Smoothness Penalty) ===
        prev_action = self._last_action.get(agent, np.zeros(2))
        curr_action = self._current_action.get(agent, np.zeros(2))
        smooth_penalty = -0.1 * np.linalg.norm(curr_action - prev_action)

        # === (5) 성공 스텝 시간 보너스 ===
        # 성공 스텝 조건 수정: 5.0m <= dist <= 10.0m로 경계 포함
        is_success_step = 1 if 5.0 <= dist <= 10.0 else 0
        self.success_steps[agent] += is_success_step
        time_bonus = 0.001 * self.success_steps[agent]

        # === (6) 최종 합산 ===
        final_reward = dist_reward + smooth_penalty + time_bonus
        
        # 비종료 스텝이므로 done=False 반환
        return float(final_reward), False
    

    # 팀 보상 집계(평균) + 안전 게이트
    def _team_reward_and_done(self, per_agent_results, mission_accomplished=False):
        # 안전 게이트: 한 명이라도 종료이면 즉시 실패
        any_fail = any(done_i for (_, done_i) in per_agent_results)
        if any_fail:
            print("🛑에이전트 하나 종료됨. 미션 실패🛑")
            return -200.0, True, {"final_status": "FAIL_CRASH"}

        # 미션 성공: 유인기가 목적지 도착
        if mission_accomplished:
            tracking_rewards = [r_i for (r_i, _) in per_agent_results]
            final_reward = float(np.mean(tracking_rewards)) + 500.0
            print("⭐️미션 성공⭐️")
            return final_reward, True, {"final_status": "SUCCESS"}

        # 미션 진행 중: 추종 보상의 평균
        tracking_rewards = [r_i for (r_i, _) in per_agent_results]
        return float(np.mean(tracking_rewards)), False, {}

    # ======================================================================
    # PettingZoo API-ish
    # ======================================================================   
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._setup_flight()
        self._generate_leader_waypoints()
        self.current_waypoint_idx = 0
        self._teleport_to_start() # <- 드론 위치가 여기서 설정됨
        self.client.simFlushPersistentMarkers()

        # ★ 에피소드 시작 시 버퍼 리셋 (기존 코드)
        self._timestep = 0
        self._last_pose.clear()
        self._last_time.clear()
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}
        self._cluster_tracks = {a: {} for a in self.possible_agents}
        self._next_cluster_id = {a: 0 for a in self.possible_agents}

        # 'step' 함수에서처럼 'now_t'와 'all_poses'를 생성합니다.
        now_t = self._timestep * self.dt  # (now_t = 0.0)
        
        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1") # 유인기
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent) # 에이전트

        obs_list = [self._get_obs(a, now_t, all_poses) for a in self.agents]
        
        return obs_list

    def step(self, actions):
        self._timestep += 1
        now_t = self._timestep * self.dt

        # A) LiDAR 기반 트랙 갱신
        for agent in self.agents:
            if agent.startswith("Follower"):
                self._update_clusters_and_tracks(agent, now_t)

        # B) 액션 적용
        for agent, act in zip(self.agents, actions):
            self._do_action(agent, act)

        # C) 리더 이동/시각화
        mission_accomplished = self._update_leader_movement()

        # ★★★ [최적화] 모든 위치 정보를 이 시점에 "한 번만" 가져옴 ★★★
        all_poses = {}
        all_poses["Drone1"] = self._get_pose_xyz("Drone1") # 유인기
        for agent in self.agents:
            all_poses[agent] = self._get_pose_xyz(agent) # 에이전트

        # D) 관측/보상/종료 신호
        obs_list, per_agent_results, per_agent_infos = [], [], []
        for agent in self.agents:
            # ★ (수정) 최적화된 _get_obs 함수 호출 ★
            o = self._get_obs(agent, now_t, all_poses)
            
            # ★ (수정) 최적화된 _compute_reward 함수 호출 ★
            r, done_i = self._compute_reward(agent, all_poses) 
            
            obs_list.append(o)
            per_agent_results.append((float(r), bool(done_i)))
            per_agent_infos.append({"reward": float(r)})

        # --- (이하 로직은 기존과 동일) ---
        team_reward, done_all, final_team_info = self._team_reward_and_done(per_agent_results, mission_accomplished)
        
        # ... (이하 동일) ...
        n = len(self.agents)
        rewards_list = [team_reward] * n
        dones_list = [done_all] * n
        infos_list = []
        for i in range(n):
            info_i = per_agent_infos[i].copy()
            info_i.update(final_team_info)
            infos_list.append(info_i)

        return obs_list, rewards_list, dones_list, infos_list
