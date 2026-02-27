import gymnasium as gym
import numpy as np
import airsim
import math
import time
import random

"""
문제점
 [ ] 무인기 움직임
    - action 공간 재정의..? (movebyvelocity냐 다른거냐, 아님 액션 공간을 줄이냐)
    - 드론 각도 제한 (세팅 json 에서 안됨)

 [x] 오브젝트 순간이동 안함. (우리가 생각하는대로 순간이동은 그냥 불가능하다고 판단된다.)
 [ ] 유인기 움직임 구현
 [ ] 유-동 충돌시 에피소드 종료 안됨
 [ ] 시각화 최적화

- 

"""

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
        far_cutoff=60.0,                        
        too_close=0.5,                          
        dt=0.01,                                
        do_visualize=True                       
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]

        # 충돌 관련 설정
        self.COLLISION_THRESHOLD = 1.0 # 모든 거리 기반 충돌 판단 임계값 (m)
        self.STOP_DISTANCE_LEADER_OBSTACLE = 1.0 # 유인기-장애물 충돌 임계값 (m)
        
        # 속도/액션 버퍼
        self.vmax_self = 2.0             
        self._timestep = 1.0

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}
        self._last_time = {}

        # 액션 버퍼
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
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

        ## 동적 장애물 관련
        self.isIdle = None
        self.D_O_STATE = { 0: "idle", 1: "attack" }
 
        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1 # 나를 제외한 아군 수
        self.K_enemy = 1                     # 동적 장애물 1대
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy
        

        low_bearing = -1.0
        high_bearing = 1.0
        low_dist = 0.0
        high_dist = 200.0
        
        # [리더와의 상대 방위, 리더와의 상대 거리], [아군 상대방위, 아군 상대 거리] * 2, [동적 장애물의 상대 방위, 동적 장애물의 상대 거리] * k
        per_agent_low = [low_bearing, low_dist] + [low_bearing, low_dist] * self.num_ally + [low_bearing, low_dist] * self.num_enemy
        per_agent_high = [high_bearing, high_dist] + [high_bearing, high_dist] * self.num_ally + [high_bearing, high_dist] * self.num_enemy

        obs_dim = len(per_agent_low)
        share_obs_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low, dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,), dtype=np.float32
            ) for agent in self.possible_agents
        }


        self.MAX_YAW   = 180
        self.MAX_PITCH = 13.0
        # Action 공간 구성 (Action[0]: Yaw Rate (회전), Action[1]: Pitch Angle (상하 기울기))
        self.action_spaces = {
            agent: gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2, ),
                dtype=np.float64
            ) for agent in self.possible_agents
        }

        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,), dtype=np.float32
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
        src_yaw = airsim.utils.to_eularian_angles(self.current_location[src_drone].orientation)[2]    # Yaw 라디안

        # 상대 거리 구하기 (피타고라스, World Frame 기준)
        distance_diff = math.sqrt(dx**2 + dy**2)

        # 상대 방위 구하기
        ## - World Frame 기준 두 좌표의 상대 방위 구하기 (arctan 활용)
        _angle = math.atan2(dx, dy)

        ## - 두 방위를 빼면 드론 기준 상대 방위를 할 수 있다. (절대 방위 차 - 현재 드론의 방위)
        angle_diff = ((_angle - src_yaw) + math.pi) % (2 * math.pi) - math.pi   # 각도 정규화 (-180, +180)으로 정규화
    
        return angle_diff, distance_diff

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

        # - moveToPositionAsync의 특성을, 원래 위치로 돌아감을 구현 -> 모든 드론은 (0.0, 0.0)으로 복귀 시킴 
        # 위치로 이동
        _command = []   # Init
        ## - 유인기
        _command.append(
            self.client.moveToPositionAsync(
                x=0.0,
                y=0.0,
                z=self.fixed_z,
                velocity=10.0,
                vehicle_name="Drone1"
            )
        )
        ## - 무인기
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
        ## - 동적 장애물 
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

        # 안정화
        time.sleep(2.0)

    def _update_leader_movement(self):
        """
        유인기에 아무런 명령도 내리지 않고, 그대로 유지, 유인기는 초기 setup시에 해당 고도로 이동한 후, 아무런 명령없이 대기
        """

        # 2. 시각화는 그대로 유지
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
    def _formation_reward(self, agent_pos, leader_pos):
        rel = leader_pos - agent_pos
        dist = math.hypot(float(rel[0]), float(rel[1]))
        if dist < 0.5 or dist > 60.0:
            return -5.0
        ideal = 10.0
        sigma = 10.0
        r = 3.0 * math.exp(-((dist - ideal) ** 2) / (2.0 * sigma ** 2)) - 1.0
        return r

    def _guardian_reward(self, agent_pos, leader_pos, dynamic_pos):
        d_lo = np.linalg.norm(leader_pos[:2] - dynamic_pos[:2])
        d_ao = np.linalg.norm(agent_pos[:2] - dynamic_pos[:2])

        ALERT_DIST = 80.0
        if d_lo > ALERT_DIST:
            return 0.0

        if d_ao < d_lo:
            score = (d_lo - d_ao) / max(d_lo, 1e-3)
            return 2.0 * score
        else:
            return -0.5

    def _compute_reward(self, agent, distance_leader, distance_other, distance_dynamic):

        # 필요한 위치들 연산 (World Frame)
        leader_pos = np.array([
            self.current_location["Drone1"].position.x_val,
            self.current_location["Drone1"].position.y_val,
            self.current_location["Drone1"].position.z_val,
        ])
        agent_pos = np.array([
            self.current_location[agent].position.x_val,
            self.current_location[agent].position.y_val,
            self.current_location[agent].position.z_val,
        ])
        dynamic_pos = np.array([
            self.current_location[self.dynamic_name].position.x_val,
            self.current_location[self.dynamic_name].position.y_val,
            self.current_location[self.dynamic_name].position.z_val,
        ])

        # 1) 유인기에 너무 가까움 → 큰 패널티 + 종료 (거리 기반)
        pass

        # 3) 포메이션 보상
        r_form = self._formation_reward(agent_pos, leader_pos)

        # 4) 가디언 위치 보상
        r_guard = self._guardian_reward(agent_pos, leader_pos, dynamic_pos)

        r_total = r_form + r_guard
        return float(r_total), False
    
    def _end_episode(self, reward, status):
        """
        에피소드 종료 헬퍼 (충돌 이벤트 발생 시)
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

        # ==========================================
        # STATE: IDLE (대기 상태)
        # ==========================================
        if self._obstacle_state == "IDLE":
            # 호버링 유지 (위치 고정)
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name)

            # [조건 체크] 정해진 대기 스텝(10~30)이 지났는가?
            if self._obs_step_timer >= self._idle_wait_steps:
                print(f"[DynamicObstacle] {self._obs_step_timer} steps passed. IDLE -> ATTACK!")
                self._obstacle_state = "ATTACK"
                self._obs_step_timer = 0  # 타이머 리셋
        
        # ==========================================
        # STATE: ATTACK (추적 상태)
        # ==========================================
        elif self._obstacle_state == "ATTACK":
            # 1. 유인기 방향 계산
            try:
                leader_pos = self.current_location["Drone1"].position
                obs_pos = self.current_location[self.dynamic_name].position
                
                l_vec = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val])
                o_vec = np.array([obs_pos.x_val, obs_pos.y_val, obs_pos.z_val])
                
                diff = l_vec - o_vec
                dist = np.linalg.norm(diff)
                
                if dist > 0.5:
                    direction = diff / dist
                    speed = 5.0  # 공격 속도 (m/s)
                    vel = direction * speed
                    
                    # 2. 속도 명령 전송 (유도탄 처럼 추적)
                    self.client.moveByVelocityAsync(
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]), 
                        duration=0.1, 
                        vehicle_name=self.dynamic_name
                    )
            except Exception as e:
                print(f"Attack Logic Error: {e}")

            # 3. [안전 장치] 너무 오랫동안(예: 200스텝) 못 맞추면 강제 리셋 (무한 추적 방지)
            if self._obs_step_timer > 200:
                 print("[DynamicObstacle] Attack Timeout. Forcing Reset.")
                 self._reset_obstacle_logic()

    def _teleport_obstacle_randomly(self):
        """장애물을 리더 근처 랜덤 위치로 순간이동 시킴"""
        leader_pos = self.client.simGetObjectPose("Drone1").position
        lx, ly = leader_pos.x_val, leader_pos.y_val
        
        # 50m ~ 60m 반경 내 랜덤 위치
        radius = random.uniform(50.0, 60.0)
        angle = random.uniform(0, 2 * math.pi)
        
        tx = lx + radius * math.cos(angle)
        ty = ly + radius * math.sin(angle)
        tz = self.fixed_z

        # 위치 강제 설정
        pose = airsim.Pose(airsim.Vector3r(tx, ty, tz), airsim.Quaternionr(0,0,0,1))
        self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.dynamic_name)
        
        # 중요: 순간이동 후 이전 속도(관성) 제거
        self.client.moveByVelocityAsync(0,0,0, duration=0.1, vehicle_name=self.dynamic_name).join()

    def _reset_obstacle_logic(self):
        """
        공격 완료/실패 후 호출:
        1. 랜덤 위치로 순간이동
        2. 상태를 IDLE로 변경
        3. 랜덤 대기 시간(10~30 step) 재설정
        """
        #self._teleport_obstacle_randomly()
        
        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(10, 30) # 다음 대기 시간 랜덤 설정
        
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

        # 각 피쳐 (리더와의, 아군과의, 장애물과의) 초기화
        _leader_feats = []
        _ally_feats = []
        _dynamic_feats = []

        # - [리더와의 상대 방위, 리더와의 상대 거리] 구하기
        _leader_feats = [self._angle_and_distance(agent, "Drone1")]

        # - [아군 상대방위, 아군 상대 거리] * 2 구하기
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            _ally_feats.append(self._angle_and_distance(agent, other))

        # - [동적 장애물의 상대 방위, 동적 장애물의 상대 거리] * K 구하기
        _dynamic_feats = [self._angle_and_distance(agent, self.dynamic_name)]

        # [리더와의 상대 방위, 리더와의 상대 거리], [아군 상대방위, 아군 상대 거리] * 2, [동적 장애물의 상대 방위, 동적 장애물의 상대 거리], 
        # 총 8개의 공간
        obs = np.concatenate([
            np.array(_leader_feats),
            np.array(_ally_feats),
            np.array(_dynamic_feats),
        ]).flatten().astype(np.float64)

        return obs

    def _do_action(self, actions):
        # Action 공간
        # (Action[0]: Yaw Rate (회전), Action[1]: x Force (좌우 가속), Action[1]: y Force (상하 가속))
        actions = np.clip(actions, -1.0, 1.0)
    
        # moveByRollPitchYawrateZAsync: 드론의 Roll(좌우 기울기), Pitch(전후 기울기), Yaw Rate(회전 속도)를 제어하면서 고도(Z)를 유지하는 함수
        # 해당 들어가는 값은 라디안 값임 따라서 받는 값은 -1 ~ 1의 값을 원하는 형태로 가공해야 함.
        # 해당 드론의 
        # - Yaw는 -180 ~ 180도까지
        # - Pitch는 한번에 13도까지

        # 변환
        for i, agent in enumerate(self.possible_agents):
            _yaw = math.radians(actions[i][0] * self.MAX_YAW)
            _pitch = math.radians(actions[i][1] * self.MAX_PITCH)
            self.client
            self.client.moveByRollPitchYawrateZAsync(
                roll=0.0,   # roll은 고정
                #pitch=_pitch,
                pitch = 0,
                yaw_rate=_yaw,
                z=self.fixed_z,
                duration=1.0,
                vehicle_name=agent
            )
    
    def _get_rewards(self, per_agent_results):
        return [np.mean(per_agent_results) for _ in self.possible_agents]

    def reset(self, seed=None, options=None):

        self.episode_count += 1
        print(f"Current Episode: {self.episode_count}")
        
        self.agents = self.possible_agents[:]

        # 위치 초기화
        self.client.reset()
        self._setup_flight()
        self.client.simFlushPersistentMarkers()
        '''        
        # FSM 상태 변수 초기화
        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(50, 100)
        '''
        #동적 장애물을 리더 근처 랜덤 위치로 초기화 
        self._reset_obstacle_logic()

        # 각 드론 위치 다시 받아오기
        self._get_current_location()
        
        # 스텝 카운트 초기화
        self.step_count = 0

        self._last_pose.clear()
        self._last_time.clear()
        self._last_action = {a: np.zeros(2, dtype=np.float32) for a in self.possible_agents}
        self.leader_stop = False

        obs_list = [self._get_obs(a) for a in self.agents]

        print("reset.")

        return obs_list

    def step(self, actions):

        # ===== 스텝 시 초기화 인스턴스들 ===== 
        # - 스텝 카운드 +1
        self.step_count += 1
        # - 에이전트 별 | 관측 / 보상 / 정보 | 인덱스 초기화
        per_agent_obs, per_agent_results, per_agent_infos = [], [], []


        # ===== Action Step =====
        # - 에이전트 액션 적용 (에이전트별 action 구현)
        self._do_action(actions)

        # - 유인기/장애물 이동
        self._update_leader_movement()
        # self._update_dynamic_obstacle()

        # 현재 위치 값 받아오기 (World Frame)
        self._get_current_location()
        #print(self.current_location["Follower1"].orientation)

        # ===== Check Termination Step =====
        for agent in self.possible_agents:
            
            # 이번 스텝에 활용할 거리 정보 데이터 미리 연산 해두기 (각 에이전트 별 기준)
            other_agents = [a for a in self.possible_agents if a != agent]  # 본인이 아닌 다른 에이전트 배열 불러오기  

            ## - 유인기와의 거리
            _distance_leader = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location["Drone1"].position.x_val,
                self.current_location[agent].position.y_val - self.current_location["Drone1"].position.y_val,
                self.current_location[agent].position.z_val - self.current_location["Drone1"].position.z_val
            ])
            ## - 다른 에이전트들 간의 거리 (List)
            _distance_other = [
                np.linalg.norm([
                    self.current_location[agent].position.x_val - self.current_location[other].position.x_val,
                    self.current_location[agent].position.y_val - self.current_location[other].position.y_val,
                    self.current_location[agent].position.z_val - self.current_location[other].position.z_val
                ]) for other in other_agents
            ]
            ## - 동적 장애물과의 거리
            _distance_dynamic = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location[self.dynamic_name].position.x_val,
                self.current_location[agent].position.y_val - self.current_location[self.dynamic_name].position.y_val,
                self.current_location[agent].position.z_val - self.current_location[self.dynamic_name].position.z_val
            ])

            # - 만약 에이전트가 유인기의 범위를 벗어났을 경우,
            if _distance_leader > self.far_cutoff:
                print(f"[이탈] {agent}가 리더와의 거리({_distance_leader:.2f}m)로, 이탈 임계값({self.far_cutoff}m) 초과! → 전체 실패")
                # -1000.0의 큰 패널티를 부여하고 에피소드 종료
                return self._end_episode(-1000.0, "FAIL_AGENT_FAR_CUTOFF")
            
            # - 만약 에이전트가 유인기와 충돌했을 경우,
            collisionInfo = self.client.simGetCollisionInfo(vehicle_name=agent)
            #print(f"leader_collision: {collisionInfo}")
            if collisionInfo.has_collided and collisionInfo.object_name == "Drone1":
                print(f"[충돌] {agent}가 리더와 충돌로 → 전체 실패")
                return self._end_episode(-1000.0, "FAIL_AGENT_AND_LEADER_COLLISION")
            
            # - 만약 에이전트가 에이전트와 충돌했을 경우,
            collisionInfo = self.client.simGetCollisionInfo(vehicle_name=agent)
            #print(f"agent_collision: {collisionInfo}")
            if collisionInfo.has_collided and (collisionInfo.object_name in other_agents):
                print(f"[충돌] {agent}가 {collisionInfo.object_name}와 충돌로 → 전체 실패")
                return self._end_episode(-1000.0, "FAIL_AGENT_AND_OTHER_AGENT_COLLISION")
            
            # - 만약 유인기가 동적장애물과 충돌했을 경우,
            collisionInfo = self.client.simGetCollisionInfo("Drone1")
            #print(f"leader_to_dyobs_collision: {collisionInfo}")
            if collisionInfo.has_collided and collisionInfo.object_name == self.dynamic_name:
                print(f"[충돌] 유인기가 {collisionInfo.object_name}와 충돌로 → 전체 실패")
                return self._end_episode(-1000.0, "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION")

            # - 만약 에이전트가 동적 장애물과 충돌했을 경우,
            collisionInfo = self.client.simGetCollisionInfo(vehicle_name=agent)
            #print(f"dynamic_collision: {collisionInfo}")
            if collisionInfo.has_collided and collisionInfo.object_name == self.dynamic_name:
                print(f"[충돌] {agent}가 동적 장애물과 충돌로 → 전체 성공 및 종료")
                return self._end_episode(500.0, "SUCCESS_AGENT_AND_DYNAMIC_OBSTACLE_COLLISION")


            # - 종료 조건을 만족하지 못한 경우,
            per_agent_obs.append(self._get_obs(agent))

            _reward, _ = self._compute_reward(agent, _distance_leader, _distance_other, _distance_dynamic)
            per_agent_results.append(_reward)
            per_agent_infos.append([f"reward: {_reward}"])

        # 도중에 종료 안되면 다 종료 안함.
        termination_list = [False for _ in self.possible_agents]

        # ===== Rewards Step =====
        rewards_list = self._get_rewards(per_agent_results)

        
        # ===== Observations Step =====
        obs_list = per_agent_obs


        # ===== Infos Step =====
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list