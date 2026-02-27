import gymnasium as gym
import numpy as np
import airsim
import math
import time
import random

class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",                 # 기본값
        follower_names=("Follower0", "Follower1", "Follower2"),
        port=41451,
        step_length=0.01,
        leader_velocity=1.0,
        optimal_distance=10.0,
        far_cutoff=30.0,                        # 유인기와 30m 이상 떨어지면 패널티 받는 구간
        too_close=0.5,
        dt=0.05,
        do_visualize=True,
        max_step=1000
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]
        # 기본 변수 설정
        self.max_step = max_step  # max_step 추가
        self.step_count = 0
        self.episode_count = 0

        # 충돌 관련 설정
        self.COLLISION_THRESHOLD = 1.0  # 모든 거리 기반 충돌 판단 임계값 (m)
        self.STOP_DISTANCE_LEADER_OBSTACLE = 1.0  # 유인기-장애물 충돌 임계값 (m)

        # 속도/액션 버퍼
        self.vmax_self = 2.0
        self._timestep = 0.05

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}
        self._last_time = {}

        # 액션 버퍼
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        # 🔹 에이전트별 "이전 스텝에서의 적군까지 거리" 저장용 버퍼
        self.prev_distance_dynamic = {a: None for a in self.possible_agents}

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

        ## 동적 장애물 관련
        self.isIdle = None
        self.D_O_STATE = {0: "idle", 1: "attack"}

        # 동적 장애물 FSM 상태 변수 초기화
        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(10, 30)

        ### 리더-동적 장애물 거리 패널티용 하이퍼파라미터
        self.leader_safe_radius = 20.0          # [m] 안전 반경
        self.max_leader_dynamic_penalty = 50.0  # 안전반경 안에서 최대 부과되는 추가 패널티(음수)

        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1  # 나를 제외한 아군 수
        self.K_enemy = 1                      # 동적 장애물 1대
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy

        # type_flag: 0.0 = 리더, +1.0 = 아군, -1.0 = 적군
        self.TYPE_LEADER = 0.0
        self.TYPE_ALLY = 1.0
        self.TYPE_ENEMY = -1.0

        low_bearing = -1.0
        high_bearing = 1.0
        low_dist = 0.0
        high_dist = 200.0
        low_type = -1.0
        high_type = 1.0

        # 한 타겟당 [bearing, dist, type_flag]
        per_target_low = [low_bearing, low_dist, low_type]
        per_target_high = [high_bearing, high_dist, high_type]

        # 타겟은 [리더 1개 + 아군 num_ally개 + 적군 num_enemy개]
        num_targets = 1 + self.num_ally + self.num_enemy +1
        per_agent_low = per_target_low * num_targets
        per_agent_high = per_target_high * num_targets

        obs_dim = len(per_agent_low)
        share_obs_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low, dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,),
                dtype=np.float32
            ) for agent in self.possible_agents
        }

        self.MAX_YAW = 180
        self.MAX_PITCH = 13.0
        # Action 공간 구성 (Action[0]: Yaw Rate (회전), Action[1]: Pitch Angle (상하 기울기))
        self.action_spaces = {
            agent: gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float64
            ) for agent in self.possible_agents
        }

        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,),
            dtype=np.float32
        )

        self.dynamic_name = "DynamicObstacle"

        # Setting json의 초기 시작 위치 및 에피소드 초기화 시 초기 위치 (World 프레임 기준)
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

        # 해당 에이전트의 방위 구하기
        src_yaw = airsim.utils.to_eularian_angles(
            self.current_location[src_drone].orientation
        )[2]  # Yaw 라디안

        # 상대 거리 구하기 (피타고라스, World Frame 기준)
        distance_diff = math.sqrt(dx ** 2 + dy ** 2)

        # 상대 방위 구하기
        _angle = math.atan2(dy, dx)
        angle_diff = ((_angle - src_yaw) + math.pi) % (2 * math.pi) - math.pi  # (-pi, +pi)
        angle_norm = angle_diff / math.pi  # [-1, 1]

        return angle_norm, distance_diff

    def _angle_dist_and_type(self, src, tgt, type_flag):
        bearing, dist = self._angle_and_distance(src, tgt)
        return [bearing, dist, type_flag]

    def _get_current_location(self):
        self.current_location = {}  # Init
        self.current_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.current_location[agent] = self.client.simGetObjectPose(agent)
        self.current_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

    # ======================================================================
    # 초기화/이동/시각화 관련
    # ======================================================================
    def _setup_flight(self):

        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")

        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
            self.client.armDisarm(True, vehicle_name=agent)

        self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
        self.client.armDisarm(True, vehicle_name=self.dynamic_name)

        # 이륙 명령 생성
        _command = []
        _command.append(self.client.takeoffAsync(vehicle_name="Drone1"))
        for agent in self.possible_agents:
            _command.append(self.client.takeoffAsync(vehicle_name=agent))
        _command.append(self.client.takeoffAsync(vehicle_name=self.dynamic_name))

        for c in _command:
            c.join()

        self.start_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.start_location[agent] = self.client.simGetObjectPose(agent)
        self.start_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

        # (0,0,fixed_z)로 복귀
        _command = []   # Init
        _command.append(
            self.client.moveToPositionAsync(
                x=0.0,
                y=0.0,
                z=self.fixed_z,
                velocity=10.0,
                vehicle_name="Drone1"
            )
        )
        for agent in self.possible_agents:
            _command.append(
                self.client.moveToPositionAsync(
                    x=0.0,
                    y=0.0,
                    z=self.fixed_z,
                    velocity=10.0,
                    vehicle_name=agent
                )
            )
        _command.append(
            self.client.moveToPositionAsync(
                x=0.0,
                y=0.0,
                z=self.fixed_z,
                velocity=10.0,
                vehicle_name=self.dynamic_name
            )
        )

        for c in _command:
            c.join()

        time.sleep(2.0)

    def _update_leader_movement(self):
        """
        유인기는 현재 정지(hover) 상태 유지
        """
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
    def _compute_reward(self, agent, distance_leader, distance_other,
                        distance_dynamic, distance_dynamic_prev,
                        bearing_dynamic, collided_with_obstacle,
                        dist_leader_dynamic,
                        collided_with_agent):
        """
        에이전트별 보상 계산

        1. 유인기와 거리 유지 보상
           - d <= 5m      : -40
           - 5 < d <= 20m : +100
           - d > 20m      : -100 (실제로는 step()에서 far_cutoff 초과 시 에피소드 종료)

        2. 가디언(요격) 시야 보상
           - 적군을 정면(상대 방위각 0)에 두면 보상 최대(+20),
             측면/후방일수록 줄어듦

        3. 적군과의 거리 보상 (절대 거리)
           - 적군에 가까워질수록 보상(+30까지)

        3-1. 스텝마다 "접근한 정도" 보상 (delta 거리 기반)
           - 이전 스텝보다 적군에게 더 가까워지면 추가 보상(+),
             멀어지면 소량 페널티(-)

        4. 에이전트가 적군과 충돌 시 +1000 보너스 (요격 성공)

        5. 유인기-동적 장애물 거리 페널티
           - 리더 근처로 적기가 들어올수록 추가 페널티

        6. 에이전트-에이전트 충돌 페널티 (에피소드 종료 없이 매 스텝 -40)
        """

        # 1) 유인기와 거리 유지 보상
        d = distance_leader
        if d <= 1.0:
            r_dist = -40.0
        elif d <= 30.0:
            r_dist = 5.0
        else:
            r_dist = -100.0  # 실제로는 far_cutoff에서 에피 종료
        '''
        # 2) 가디언 시야 보상 (적군을 정면에 둘수록 보상)   => 이거를 없애보자 !
        if bearing_dynamic is not None:
            # bearing_dynamic은 이미 [-1, 1] 범위
            norm_abs = min(abs(bearing_dynamic), 1.0)  # 0(정면) ~ 1(반대 방향)
            r_bearing = 20.0 * (1.0 - norm_abs)       # 정면일 때 +20, 옆/뒤로 갈수록 0
        else:
            r_bearing = 0.0
        '''

        # 3) 적군과의 거리 보상 (가까울수록 보상)
        max_enemy_reward_dist = 50.0
        clamped = min(distance_dynamic, max_enemy_reward_dist)
        r_enemy_dist = 30.0 * (1.0 - clamped / max_enemy_reward_dist)  # 0m: +30, 50m: 0

        # 3-1) 스텝마다 "접근한 정도" 보상 (delta 거리 기반)
        r_enemy_step = 0.0
        if distance_dynamic_prev is not None:
            delta = distance_dynamic_prev - distance_dynamic  # +면 가까워짐, -면 멀어짐
            # 폭주 방지용 clamp
            delta = max(min(delta, 5.0), -5.0)  # -5m ~ +5m 범위만 반영

            if delta > 0:
                # 가까워졌을 때 강한 보상
                r_enemy_step = 10.0 * delta    # 1m 접근 시 +10
            else:
                # 멀어졌을 때는 약한 페널티
                r_enemy_step = 5.0 * delta     # 1m 멀어지면 -5

        # 4) 적군과 충돌(요격) 시 보너스
        r_collision_enemy = 100.0 if collided_with_obstacle else 0.0

        # 5) 유인기-동적 장애물 거리 기반 페널티
        if dist_leader_dynamic < self.leader_safe_radius:
            # closeness: 0 (경계) ~ 1 (완전 겹침)
            closeness = (self.leader_safe_radius - dist_leader_dynamic) / self.leader_safe_radius
            closeness = max(0.0, min(1.0, closeness))
            r_leader_safety = - self.max_leader_dynamic_penalty * closeness
        else:
            r_leader_safety = 0.0

        # 6) 에이전트-에이전트 충돌 페널티 (에피 종료 없이 매 스텝 부과) => 지금 에이전트와 에이전트 충돌 회피 보상 아예 없애 놓음
        #r_agent_collision = -40.0 if collided_with_agent else 0.0
        # + r_agent_collision
        r_total = (
            r_dist
            + r_enemy_dist
            + r_enemy_step
            + r_collision_enemy
            + r_leader_safety
        )
        return float(r_total)

    def _end_episode(self, reward, status):
        """
        에피소드 종료 헬퍼 (충돌 이벤트/이탈 등 발생 시)
        - 현재 구현은 모든 에이전트에게 동일한 종료 보상을 부여
        """
        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        for agent in self.possible_agents:
            _obs_list.append(self._get_obs(agent))
            _rewards_list.append(reward)
            _terminations_list.append(True)
            _infos_list.append({agent: {"final_status": status, "reward": reward}})

        return _obs_list, _rewards_list, _terminations_list, _infos_list

    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self):
        """
        동적 장애물 FSM (Step Count 기반)
        """
        self._obs_step_timer += 1  # 현재 상태에서의 경과 스텝 증가

        if self._obstacle_state == "IDLE":
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name)

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
                    speed = 2.0  # 공격 속도 (m/s)
                    vel = direction * speed

                    self.client.moveByVelocityAsync(
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                        duration=0.1,
                        vehicle_name=self.dynamic_name
                    )
            except Exception as e:
                print(f"Attack Logic Error: {e}")

            if self._obs_step_timer > 1000:
                print("[DynamicObstacle] Attack Timeout. Forcing Reset.")
                self._reset_obstacle_logic()

    def _teleport_obstacle_randomly(self):
        """장애물을 리더 근처 랜덤 위치로 순간이동 시킴"""
        leader_pos = self.client.simGetObjectPose("Drone1").position
        lx, ly = leader_pos.x_val, leader_pos.y_val

        radius = random.uniform(50.0, 60.0)
        angle = random.uniform(0, 2 * math.pi)

        tx = lx + radius * math.cos(angle)
        ty = ly + radius * math.sin(angle)
        tz = self.fixed_z

        pose = airsim.Pose(airsim.Vector3r(tx, ty, tz), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.dynamic_name)

        self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name).join()

    def _reset_obstacle_logic(self):
        self._teleport_obstacle_randomly()

        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(10, 30)

        print(f"[DynamicObstacle] Reset to IDLE. Waiting for {self._idle_wait_steps} steps.")

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

        leader_feats = self._angle_dist_and_type(agent, "Drone1", self.TYPE_LEADER)

        ally_feats = []
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            ally_feats.append(self._angle_dist_and_type(agent, other, self.TYPE_ALLY))

        dynamic_feats = []
        dynamic_feats.append(self._angle_dist_and_type(agent, self.dynamic_name, self.TYPE_ENEMY))

        # 리더-동적장애물 거리 관측 추가 (bearing=0, type=-1 로 처리)
        leader_pos = self.current_location["Drone1"].position
        dyn_pos = self.current_location[self.dynamic_name].position
        dist_leader_dynamic = np.linalg.norm([
            leader_pos.x_val - dyn_pos.x_val,
            leader_pos.y_val - dyn_pos.y_val,
            leader_pos.z_val - dyn_pos.z_val
        ])
        
        leader_dynamic_feat = np.array([[0.0, dist_leader_dynamic, -1.0]], dtype=np.float32)

        feats = np.concatenate([
            np.array(leader_feats, dtype=np.float32).reshape(1, -1),
            np.array(ally_feats, dtype=np.float32),
            np.array(dynamic_feats, dtype=np.float32),
            leader_dynamic_feat
        ], axis=0)

        return feats.flatten().astype(np.float32)



    def _do_action(self, actions):
        actions = np.clip(actions, -1.0, 1.0)
        futures = []
        for i, agent in enumerate(self.possible_agents):
            _yaw = math.radians(actions[i][0] * self.MAX_YAW)
            _pitch = math.radians(actions[i][1] * self.MAX_PITCH)
            f = self.client.moveByRollPitchYawrateZAsync(
                roll=0.0,
                pitch=_pitch,
                yaw_rate=_yaw,
                z=self.fixed_z,
                duration=self._timestep,
                vehicle_name=agent
            )
            futures.append(f)
        for f in futures:
            f.join()

    def _get_rewards(self, per_agent_results):
        return per_agent_results

    def reset(self, seed=None, options=None):

        self.episode_count += 1
        print(f"Current Episode: {self.episode_count}")

        self.agents = self.possible_agents[:]

        self.client.reset()
        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        self._reset_obstacle_logic()

        self._get_current_location()

        self.step_count = 0

        self._last_pose.clear()
        self._last_time.clear()
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.leader_stop = False

        # 🔹 에이전트별 초기 적군 거리 저장
        for agent in self.possible_agents:
            self.prev_distance_dynamic[agent] = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location[self.dynamic_name].position.x_val,
                self.current_location[agent].position.y_val - self.current_location[self.dynamic_name].position.y_val,
                self.current_location[agent].position.z_val - self.current_location[self.dynamic_name].position.z_val
            ])

        obs_list = [self._get_obs(a) for a in self.agents]

        print("reset.")

        return obs_list

    def step(self, actions):

        self.step_count += 1
        per_agent_obs, per_agent_results, per_agent_infos = [], [], []

        # ===== Action Step =====
        self._do_action(actions)
        self._update_leader_movement()
        self._update_dynamic_obstacle()
        self._get_current_location()

        ### 리더-동적 장애물 거리 계산 (모든 에이전트 보상에 공통 사용)
        leader_pos = self.current_location["Drone1"].position
        dyn_pos = self.current_location[self.dynamic_name].position
        dist_leader_dynamic = np.linalg.norm([
            leader_pos.x_val - dyn_pos.x_val,
            leader_pos.y_val - dyn_pos.y_val,
            leader_pos.z_val - dyn_pos.z_val
        ])

        # 유인기-동적 장애물 충돌 체크 (에피소드 종료 + 큰 패널티)
        leader_collision = self.client.simGetCollisionInfo("Drone1")
        leader_dynamic_collision = False
        if leader_collision.has_collided and leader_collision.object_name == self.dynamic_name:
            print(f"[충돌] 유인기가 {leader_collision.object_name}와 충돌")
            #return self._end_episode(-100.0, "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION")
            leader_dynamic_collision = True

        # ===== Check Termination & Reward Step =====
        for agent in self.possible_agents:

            other_agents = [a for a in self.possible_agents if a != agent]

            _distance_leader = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location["Drone1"].position.x_val,
                self.current_location[agent].position.y_val - self.current_location["Drone1"].position.y_val,
                self.current_location[agent].position.z_val - self.current_location["Drone1"].position.z_val
            ])
            _distance_other = [
                np.linalg.norm([
                    self.current_location[agent].position.x_val - self.current_location[other].position.x_val,
                    self.current_location[agent].position.y_val - self.current_location[other].position.y_val,
                    self.current_location[agent].position.z_val - self.current_location[other].position.z_val
                ]) for other in other_agents
            ]
            _distance_dynamic = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location[self.dynamic_name].position.x_val,
                self.current_location[agent].position.y_val - self.current_location[self.dynamic_name].position.y_val,
                self.current_location[agent].position.z_val - self.current_location[self.dynamic_name].position.z_val
            ])
            '''
            # 1) 유인기와의 거리 기반 종료 조건
            if _distance_leader > self.far_cutoff:
                print(f"[이탈] {agent}가 리더와의 거리({_distance_leader:.2f}m)로, 이탈 임계값({self.far_cutoff}m) 초과! → 전체 실패 및 종료")
                return self._end_episode(-100.0, "FAIL_AGENT_FAR_CUTOFF")
            '''
            collided_with_obstacle = False
            collided_with_agent = False
            collided_with_leader = False
            # 2) 에이전트/유인기/에이전트 간 충돌 체크
            collided_with_obstacle = False
            collided_with_agent = False

            collisionInfo = self.client.simGetCollisionInfo(vehicle_name=agent)
            if collisionInfo.has_collided:
                if collisionInfo.object_name == "Drone1":
                    print(f"[충돌] {agent}가 리더와 충돌")
                    #return self._end_episode(-40.0, "FAIL_AGENT_AND_LEADER_COLLISION")
                elif collisionInfo.object_name in other_agents:
                    # 🔹 에이전트-에이전트 충돌: 에피소드 종료는 하지 않고 매 스텝 패널티만 부여
                    print(f"[충돌] {agent}가 {collisionInfo.object_name}와 충돌")
                    collided_with_agent = True
                elif collisionInfo.object_name == self.dynamic_name:
                    print(f"[충돌] {agent}가 동적 장애물과 충돌 → 보상 +100, 에피소드는 계속 진행")
                    collided_with_obstacle = True

            per_agent_obs.append(self._get_obs(agent))

            bearing_dynamic, _ = self._angle_and_distance(agent, self.dynamic_name)

            #  이전 스텝에서의 적군 거리
            _prev_distance_dynamic = self.prev_distance_dynamic.get(agent, None)

            _reward = self._compute_reward(
                agent,
                _distance_leader,
                _distance_other,
                _distance_dynamic,
                _prev_distance_dynamic,      # delta 기반 보상에 사용
                bearing_dynamic,
                collided_with_obstacle,
                dist_leader_dynamic,
                collided_with_agent
            )

            # Add penalties
            if collided_with_leader:
                _reward -= 40.0  # 유인기와 충돌 시 패널티
            if leader_dynamic_collision:
                _reward -= 100.0  # 유인기-적군 충돌 시 패널티

            # Append results
            per_agent_results.append(_reward)
            per_agent_infos.append(f"reward: {_reward}")  # 수정된 구문

             #  다음 스텝을 위해 현재 적군 거리 저장
            self.prev_distance_dynamic[agent] = _distance_dynamic

        # max_step에 도달했는지 체크
        termination_list = [False for _ in self.possible_agents]
        if self.step_count >= self.max_step:
            print(f"Max step {self.max_step} reached. Ending episode.")
            termination_list = [True for _ in self.possible_agents]  # 모든 에이전트를 종료 상태로 설정

        rewards_list = self._get_rewards(per_agent_results)
        obs_list = per_agent_obs
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list

