# from MARL_test import ParallelEnv
import gym
import numpy as np
import airsim
import math
import time
import random
import open3d as o3d
from sklearn.cluster import DBSCAN

class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",                 # 기본값
        follower_names=("Follower0","Follower1","Follower2"),
        step_length=1.0,
        fixed_z=-10.0,
        leader_velocity=6.0,            # 유인기 속도(m/s) 파라미터화
        # lidar_bins=36,                  # LiDAR 섹터 수 파라미터화
        # lidar_max=20.0,                 # LiDAR 최대 감지 거리(클립)
        optimal_distance=10.0,          # 추종 최적 거리
        far_cutoff=60.0,                # 이탈 종료 거리
        too_close = 1.0,                # 유인기와 최소 거리
        dt=0.05,                        # 내부 시뮬레이션 타임스텝(초) - 리더 이동/시각화 주기
        do_visualize=True               # 원/포인트 시각화 on/off
        
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]

        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = float(fixed_z)
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)

        self.leader_velocity = float(leader_velocity)
        # self.lidar_bins = int(lidar_bins)
        # self.lidar_max = float(lidar_max)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names) 

        self._first_setup = True

        # 관측은 [dist_to_leader, relx, rely] = 3차원
        self.dist_clip_max = max(self.far_cutoff, self.optimal_distance * 3.0)  # 관측 거리 클립 상한
        # ===== obs / act / share_obs spaces =====
        obs_dim = 3 + 36   # dist_to_leader, relx, rely, lidar(36)
        act_dim = 2
        share_obs_dim = obs_dim * len(self.possible_agents)

        # per-agent observation space (39차원)
        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array([0.0, -200.0, -200.0] + [0.0]*36, dtype=np.float32),
                high=np.array([self.dist_clip_max, 200.0, 200.0] + [self.far_cutoff]*36, dtype=np.float32),
                shape=(obs_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # action space (그대로)
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # shared observation space (모든 에이전트 obs concat: 39 * num_agents)
        per_agent_low  = [0.0, -200.0, -200.0] + [0.0]*36
        per_agent_high = [self.dist_clip_max, 200.0, 200.0] + [self.far_cutoff]*36
        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,),
            dtype=np.float32
        )

        self.current_waypoint_idx = 0
        self._setup_flight()
        self._generate_leader_waypoints()
        self._last_visualize_t = time.time()


        # ===== Perception Pipeline Parameters =====
        # 1. Voxel Filtering (논문 Section 2.2)
        # 10cm 크기의 정육면체로 공간을 나눔 (튜닝 필요)
        self.voxel_size = 0.1 
        
        # 2. RANSAC Ground Removal (논문 Section 2.2)
        # 지면으로 탐지할 평면의 두께 (m)
        self.ransac_distance_threshold = 0.1 
        self.ransac_n = 3
        self.ransac_iterations = 100
        
        # 3. DBSCAN Clustering (논문 Section 3.1)
        # 클러스터로 인정할 최대 거리 (eps) (튜닝 필요)
        self.dbscan_eps = 1.0 # 1.0m
        # 클러스터를 구성할 최소 포인트 수
        self.dbscan_min_samples = 5



    # ===== 라이다 관측 =====

    def _get_lidar_obs(self, agent, lidar_name="LidarSensor1"):
        """
        1. Lidar 데이터 수집 및 병합
        2. Voxel 필터링 
        3. RANSAC 지면 제거 
        4. DBSCAN 클러스터링 
        
        반환값:
            list[np.ndarray]: 탐지된 모든 객체(클러스터)의 중심점(centroid) 리스트.
                              예: [array([x1,y1,z1]), array([x2,y2,z2]), ...]
                              (Lidar 로컬 좌표계 기준)
        """

        lidar_data = self.client.getLidarData(vehicle_name=agent, lidar_name=lidar_name)
        if len(lidar_data.point_cloud) < 3:
            return np.full(36, self.far_cutoff, dtype=np.float32)  # 감지 실패 시 max range 리턴

        try:
            np.savetxt(f"{agent}_lidar_data.txt", pts, fmt='%.8f', delimiter=' ')
        except Exception as e:
            print(f"Lidar 데이터 저장 실패: {e}")
        
        # PointCloud → (x, y, z) reshape
        pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        # XY 평면에서의 거리만 사용
        dists = np.linalg.norm(pts[:, :2], axis=1)

        # 36 방향(bin)으로 최소 거리 추출
        bins = 36
        angles = np.arctan2(pts[:,1], pts[:,0])  # -pi ~ pi
        bin_indices = ((angles + np.pi) / (2*np.pi) * bins).astype(int) % bins
        min_dists = np.full(bins, self.far_cutoff, dtype=np.float32)
        for b in range(bins):
            sel = dists[bin_indices == b]
            if len(sel) > 0:
                min_dists[b] = min(sel)

        return min_dists



        # ===== VecEnv / MADDPG 호환 프로퍼티 =====

    def perform_dbscan_clustering(point_cloud, eps=0.3, min_samples=5):
        """
        논문(Section 3.1)에서 설명된 DBSCAN 클러스터링을 수행합니다. 
        
        이 함수는 Algorithm 1의 프로세스를 구현하며,
        Figure 1의 예시 파라미터(eps=0.3, min_samples=5)를 기본값으로 사용합니다. 

        Args:
            point_cloud (np.ndarray): 전처리가 완료된 (지면이 제거된) 
                                    포인트 클라우드 (N, 3 형태의 X, Y, Z 좌표). [cite: 87, 96]
            eps (float): DBSCAN의 탐색 반경 (ε). [cite: 111]
            min_samples (int): 클러스터를 형성하기 위한 최소 포인트 수 (minPts). [cite: 111]

        Returns:
            dict: 클러스터링 결과. 
                {cluster_id: np.ndarray} 형태의 딕셔너리.
                각 값은 해당 클러스터에 속한 포인트들의 (K, 3) 배열입니다.
                (논문의 Ct = {c_t^1, ...}에 해당) 
        """
        
        print(f"DBSCAN 클러스터링 시작... (eps={eps}, min_samples={min_samples})")
        
        # 1. DBSCAN 모델 초기화 및 학습
        # 논문의 Algorithm 1 (RangeQuery, minPts 확인 등)을 수행합니다. [cite: 111]
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(point_cloud)
        
        # 2. 결과 레이블 추출
        labels = db.labels_
        
        # 3. 클러스터 분리
        clusters = {}
        unique_labels = set(labels)
        
        # -1은 DBSCAN에서 '노이즈'로 분류된 포인트를 의미합니다. [cite: 103, 108]
        # 노이즈를 제외한 실제 클러스터만 추출합니다.
        unique_labels.discard(-1) 
        
        for label in unique_labels:
            # 해당 레이블(ID)을 가진 포인트들만 선택
            cluster_points = point_cloud[labels == label]
            clusters[label] = cluster_points
            
        print(f"클러스터링 완료. 노이즈 제외 총 {len(clusters)}개의 객체(클러스터) 발견.")
        
        # 4. 객체(클러스터) 딕셔너리 반환
        return clusters

    # --- 이 함수를 사용하는 방법 (예시) ---
    if __name__ == '__main__':
        
        # [가정] 
        # 1. client.getLidarData()로 원본 포인트를 받았습니다.
        # 2. Voxel 필터링을 수행했습니다. 
        # 3. RANSAC으로 지면을 제거했습니다. 
        # 'preprocessed_obstacle_points'는 이 모든 과정이 끝난 결과물입니다.
        
        # (임시 예제 데이터 생성)
        # 실제로는 AirSim에서 받은 필터링된 데이터를 사용해야 합니다.
        np.random.seed(42)
        dummy_obstacle1 = np.random.rand(100, 3) + np.array([5, 5, 0]) # 100개 포인트
        dummy_obstacle2 = np.random.rand(80, 3)  + np.array([10, -3, 0]) # 80개 포인트
        dummy_noise = np.random.rand(50, 3) * 20 # 노이즈
        preprocessed_obstacle_points = np.vstack([dummy_obstacle1, dummy_obstacle2, dummy_noise])

        
        # [DBSCAN 실행]
        # 논문의 Figure 1에서 사용된 파라미터로 클러스터링 수행 
        detected_objects = perform_dbscan_clustering(
            preprocessed_obstacle_points, 
            eps=0.5, # (예제 데이터에 맞게 0.3 -> 0.5로 조정)
            min_samples=5
        )
        
        # [결과 확인]
        # detected_objects는 딕셔너리 -> {0: array(...), 1: array(...)}
        for obj_id, points in detected_objects.items():
            centroid = np.mean(points, axis=0)
            print(f"  -> 객체 ID {obj_id}: {len(points)}개 포인트. 중심점: {centroid.round(2)}")
            
        # 'detected_objects' 딕셔너리를 다음 단계인
        # '동적/정적 분류' (Section 3.2) 알고리즘의 입력으로 사용합니다.
    


    @property
    def observation_space(self):
        # per-agent 리스트 (순서: self.possible_agents)
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

 # --------------------- 초기화/이륙 ---------------------
    def _setup_flight(self):
        if self._first_setup:
            self.client.reset()
            # 리더
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

        # API 제어 보장 (세션 중 끊겼을 수 있음)
        self.client.enableApiControl(True, vehicle_name="Drone1")
        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)

        # 텔레포트 (ignore_collision=True로 안전하게 순간이동)
        for name, (x, y, z) in start_cfg.items():
            px, py, pz = float(x), float(y), float(z)
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(px, py, pz),
                            airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)),
                ignore_collision=True,
                vehicle_name=name
            )
        time.sleep(0.05)  # 짧은 안정화 대기

    

 # --------------------- 유인기 경로 생성 ---------------------
    def _generate_leader_waypoints(self):
        """시작점 근처에서 무작위 최종 목적지를 생성하고, 깃발 객체를 그 위치로 이동시킵니다."""
        # 유인기의 시작 위치를 기준으로 경로 생성
        leader_start_pos = np.array([5.0, 2.5, self.fixed_z])
        
        # 시작점으로부터 50~70m 떨어진 곳에 무작위 목적지 설정
        distance = random.uniform(50.0, 70.0)
        angle = random.uniform(0, 2 * np.pi)
        
        # 최종 목적지 좌표 계산
        final_destination = leader_start_pos + np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            0
        ])

        # 유인기의 경로는 [최종 목적지] 하나만 가집니다.
        self.leader_waypoints = [final_destination]
        self.current_waypoint_idx = 0

        # --- 추가된 부분: 깃발 객체(목적지) 이동 ---
        try:
            # 1. 언리얼 에디터에서 설정한 깃발의 '액터 라벨'
            object_name = "target1v1_5" 

            # 2. 깃발의 위치 설정 (계산된 목적지 좌표 사용)
            # (Z좌표는 드론 고도보다 약간 높게 설정하여 잘 보이도록 함)
            flag_position = airsim.Vector3r(float(final_destination[0]),
                                            float(final_destination[1]),
                                            float(self.fixed_z + 8.0))
            
            # 3. 깃발의 방향 설정 (회전 없음)
            flag_orientation = airsim.to_quaternion(0, 80.1, 0)
            
            # 4. 위치와 방향을 합쳐 Pose 객체 생성
            flag_pose = airsim.Pose(flag_position, flag_orientation)
            
            # 5. simSetObjectPose API를 호출하여 깃발 이동
            self.client.simSetObjectPose(object_name, flag_pose)

        except Exception as e:
            print(f"'{object_name}' 객체를 이동 실패. 언리얼 레벨에 해당 이름의 객체가 있는지 확인.")
            print(e)



# --------------------- 유인기 이동 ---------------------
    def _update_leader_movement(self):
        """
        유인기를 목적지로 이동시키고, 도착 시 성공 여부를 반환합니다.
        Returns:
            mission_accomplished (bool): 유인기가 최종 목적지에 도착했는지 여부.
        """
        if not self.leader_waypoints:
            self._generate_leader_waypoints()

        # 최종 목적지
        target = self.leader_waypoints[0]

        pose = self.client.simGetVehiclePose(vehicle_name="Drone1")
        cur = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])

        # --- 목적지 도착 판정 (XY 거리 기준) ---
        dist_to_target = np.linalg.norm(target[:2] - cur[:2])
        if dist_to_target < 3.0:  # 도착 시
            return True  # 미션 성공!

        # --- 목적지를 향해 이동 (기존 로직과 유사) ---
        dir_vec = target - cur
        dist = np.linalg.norm(dir_vec[:2])
        if dist > 1e-6:
            dir_unit = dir_vec / (dist + 1e-9)
            move = dir_unit * self.leader_velocity * self.dt
            
            new_pos = cur + move
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(new_pos[0], new_pos[1], self.fixed_z), airsim.Quaternionr()),
                ignore_collision=True,
                vehicle_name="Drone1"
            )

        # --- 시각화 로직 ---
        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.1:
                # 1. 이전의 모든 마커를 깨끗하게 지웁니다.
                self.client.simFlushPersistentMarkers()
                # 2. 새로운 마커들을 그립니다.
                self._visualize_circles()
                
                self._last_visualize_t = now

        return False # 아직 미션 진행 중



# --------------------- 유인기 시각화 ---------------------
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

            # optimal, far cutoff 링
            line_thickness = 20.0
            self.client.simPlotLineStrip(ring_points(self.optimal_distance), [1, 1, 0, 0.8], line_thickness, 0.15, True)
            self.client.simPlotLineStrip(ring_points(self.far_cutoff), [0, 1, 0, 0.8], line_thickness, 0.15, True)
            
        except Exception:
            pass  # 시각화 실패는 무시

        



# --------------------- 관측 ---------------------
    def _get_obs(self, agent):
        leader_pos = self.client.simGetObjectPose("Drone1").position
        agent_pos  = self.client.simGetObjectPose(agent).position
        rel = np.array([leader_pos.x_val - agent_pos.x_val,
                        leader_pos.y_val - agent_pos.y_val], dtype=np.float32)
        dist = float(np.linalg.norm(rel))
        clipped_dist = float(np.clip(dist, 0.0, self.dist_clip_max))
        base_obs = np.array([clipped_dist, rel[0], rel[1]], dtype=np.float32)
        # obs = np.array([clipped_dist, rel[0], rel[1]], dtype=np.float32)

        # ✅ Follower만 LiDAR 관측 추가
        if agent.startswith("Follower"):
            lidar_obs = self._get_lidar_obs(agent)
            obs = np.concatenate([base_obs, lidar_obs], axis=0).astype(np.float32)
        else:
            obs = base_obs  # 리더는 LiDAR 없음

        return obs


 # --------------------- 에이전트 이동 ---------------------
    def _do_action(self, agent, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        pose = self.client.simGetVehiclePose(agent)
        x, y, z = pose.position.x_val, pose.position.y_val, self.fixed_z

        # numpy.float32 → Python float 변환
        dx = float(a[0]) * float(self.step_length)
        dy = float(a[1]) * float(self.step_length)

        nx = float(x) + dx
        ny = float(y) + dy
        nz = float(z)

        new_pos = airsim.Vector3r(nx, ny, nz)
        new_pose = airsim.Pose(new_pos, airsim.Quaternionr(0.0, 0.0, 0.0, 1.0))
        self.client.simSetVehiclePose(new_pose, False, vehicle_name=agent)

    # --------------------- 보상 ---------------------
    def _compute_reward(self, agent, obs=None):
        rewards_dict = {}

        # 1. 충돌 확인 (가장 강력한 페널티)
        # collision_info = self.client.simGetCollisionInfo(vehicle_name=agent)
        # if collision_info.has_collided:
        #     rewards_dict['penalty_collision'] = -1000.0
        #     return rewards_dict, True

        agent_pos = self.client.simGetObjectPose(agent).position
        for other in self.agents:
            if other == agent:
                continue
            other_pos = self.client.simGetObjectPose(other).position
            dist_agents = np.linalg.norm([
                agent_pos.x_val - other_pos.x_val,
                agent_pos.y_val - other_pos.y_val
            ])
            if dist_agents < 0.5:  # 1m 이내 접근 → 충돌로 간주
                rewards_dict['penalty_agent_collision'] = -1000.0
                return rewards_dict, True
            
        lidar_obs = self._get_lidar_obs(agent)
        min_dist = np.min(lidar_obs)

        if min_dist < 1.0:  # 1m 이내 근접 → 경고 페널티만
            rewards_dict['penalty_lidar_near'] = -500.0
            # done = False (아직 종료는 아님)
            return rewards_dict, False

        # 2. 거리 계산
        if obs is None:
            obs = self._get_obs(agent)
        dist_raw = math.hypot(float(obs[1]), float(obs[2]))

        # 2. 유인기와 너무 가까우면 (충돌 대신 처리)
        if dist_raw < self.too_close:
            rewards_dict['penalty_too_close'] = -1000.0
            return rewards_dict, True

        # 3. 편대 이탈 확인 (두 번째로 강한 페널티)
        if dist_raw > self.far_cutoff:
            rewards_dict['penalty_too_far'] = -1000.0
            return rewards_dict, True

        # 4. 새로운 통합 거리 보상
        sigma = 5.0                     # 곡선 폭

        if dist_raw <= self.optimal_distance:
            # 가우시안 기반 스케일링 (0m → 0, d_star → 10)
            gauss_val = math.exp(-((dist_raw - self.optimal_distance) ** 2) / (2 * sigma**2))
            gauss_at_0 = math.exp(-(self.optimal_distance**2) / (2 * sigma**2))  # d=0일 때 값

            reward_val = 10.0 * (gauss_val - gauss_at_0) / (1 - gauss_at_0 + 1e-9)

        elif dist_raw <= self.far_cutoff:
            t = (dist_raw - self.optimal_distance) / (self.far_cutoff - self.optimal_distance + 1e-9)
            alpha = 6.0  # 커질수록 오른쪽에서 더 급격히 패널티 증가 ← (그래프 느낌대로)
            # 끝점 고정: dist=d* ⇒ +10, dist=d_far ⇒ -1000
            reward_val = 10.0 + (-1010.0) * ((math.exp(alpha * t) - 1.0) / (math.exp(alpha) - 1.0))
        rewards_dict['distance_reward'] = reward_val


        return rewards_dict, False

    # --------------------- 팀 보상 집계(평균) + 안전 게이트 ---------------------
    def _team_reward_and_done(self, per_agent_results, mission_accomplished=False):
        # 안전 게이트: 한 명이라도 추락/이탈하면 즉시 실패 및 강한 페널티
        any_fail = any(done_i for (_, done_i) in per_agent_results)
        if any_fail:
            return -200.0, True, {"final_status": "FAIL_CRASH"}

        # 미션 성공: 모든 추종 드론이 살아남은 상태에서 유인기가 도착
        if mission_accomplished:
            # 기본 추종 보상 + 큰 성공 보너스
            tracking_rewards = [r_i for (r_i, _) in per_agent_results]
            final_reward = np.mean(tracking_rewards) + 500.0
            return final_reward, True, {"final_status": "SUCCESS"}

        # 미션 진행 중: 추종 보상의 평균만 계산
        tracking_rewards = [r_i for (r_i, _) in per_agent_results]
        return np.mean(tracking_rewards), False, {}



# --------------------- PettingZoo API ---------------------
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._setup_flight()
        self._generate_leader_waypoints()
        self.current_waypoint_idx = 0
        self._teleport_to_start()
        self.client.simFlushPersistentMarkers()
        # obs = {agent: self._get_obs(agent) for agent in self.agents}
        # infos = {agent: {} for agent in self.agents}
        # return obs, infos
        return [self._get_obs(a) for a in self.agents]


    def step(self, actions):
        # 1) & 2) 액션 적용 및 리더 이동 (기존과 동일)
        for agent, act in zip(self.agents, actions):
            self._do_action(agent, act)
        mission_accomplished = self._update_leader_movement()

        # 3) 관측, 개별 보상, 개별 상세 정보 수집
        obs_list = []
        per_agent_results = []
        per_agent_detailed_infos = [] # 각 에이전트의 상세 정보를 담을 리스트

        for agent in self.agents:
            o = self._get_obs(agent)
            rewards_dict, done_i = self._compute_reward(agent, obs=o)

            total_reward = sum(rewards_dict.values())
            
            obs_list.append(o)
            per_agent_results.append((total_reward, done_i))
            per_agent_detailed_infos.append({"rewards": rewards_dict})

        # 4) 팀 보상 및 최종 종료 여부, 그리고 '팀 전체'의 최종 정보 결정
        team_reward, done_all, final_team_info = self._team_reward_and_done(per_agent_results, mission_accomplished)
        
        # 5) 최종 반환값 생성
        n = len(self.agents)
        rewards_list = [team_reward] * n
        dones_list = [done_all] * n
        
        # 각 에이전트의 개별 상세 정보에 팀 전체의 최종 정보를 합쳐줍니다.
        infos_list = []
        for i in range(n):
            # 1. 개별 info 딕셔너리를 복사합니다.
            agent_info = per_agent_detailed_infos[i].copy()
            # 2. 복사한 딕셔너리에 팀 전체 info를 합칩니다.
            agent_info.update(final_team_info)
            infos_list.append(agent_info)

        return obs_list, rewards_list, dones_list, infos_list
