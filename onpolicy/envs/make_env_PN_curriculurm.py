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

        # =========================
        # ✅ [PATCH 5] action buffer shape -> (3,)
        # =========================
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

        self.stats_history = {
            "win": deque(maxlen=20),
            "coll_leader": deque(maxlen=20),
            "coll_drone": deque(maxlen=20),
            "coll_obj": deque(maxlen=20)
        }
        self.difficulty_stage = 0

        # 동적 장애물 관련
        self.isIdle = None
        self.D_O_STATE = {0: "idle", 1: "attack"}

        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1
        self.K_enemy = 1
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy

        # ---- 관측 공간 범위 정의 ----
        low_bearing = -1.0
        high_bearing = 1.0
        low_dist = 0.0
        high_dist = 200.0

        low_vel = -1.0
        high_vel = 1.0
        low_rate = -1.0
        high_rate = 1.0
        low_self_state = -1.0
        high_self_state = 1.0

        # [리더(2)] + [아군(2)*K] + [적(4)*K] + [self_state(5)]
        per_agent_low = (
            [low_bearing, low_dist] +
            [low_bearing, low_dist] * self.num_ally +
            [low_bearing, low_dist, low_vel, low_rate] * self.num_enemy +
            [low_self_state] * 5
        )
        per_agent_high = (
            [high_bearing, high_dist] +
            [high_bearing, high_dist] * self.num_ally +
            [high_bearing, high_dist, high_vel, high_rate] * self.num_enemy +
            [high_self_state] * 5
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
                shape=(3,),
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
        self.REWARD_HIT_ENEMY = -300.0
        self.REWARD_LEADER_HIT = -250.0
        self.REWARD_AGENT_CRASH = -50.0

        self.W_CLOSE = 0.5
        self.W_LOS = 0.5
        self.W_DIST = 0.5
        self.W_ALLY = 0.5
        self.STEP_PENALTY = 0.05

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
    # 헬퍼 메서드: 포즈/속도/관측 관련
    # ======================================================================
    def _angle_and_distance(self, src_drone, target_drone):
        dx = float(self.current_location[target_drone].position.x_val -
                   self.current_location[src_drone].position.x_val)
        dy = float(self.current_location[target_drone].position.y_val -
                   self.current_location[src_drone].position.y_val)

        # ✅ 실제 yaw 사용
        st = self.client.getMultirotorState(vehicle_name=src_drone)
        q = st.kinematics_estimated.orientation
        _, _, src_yaw = airsim.to_eularian_angles(q)

        distance = math.hypot(dx, dy)
        abs_bearing = math.atan2(dy, dx)
        rel_bearing = ((abs_bearing - src_yaw) + math.pi) % (2 * math.pi) - math.pi

        # =========================
        # ✅ [PATCH 2] bearing normalize: /pi -> [-1,1]
        # =========================
        rel_bearing_norm = float(np.clip(rel_bearing / math.pi, -1.0, 1.0))

        return rel_bearing_norm, distance

    # =========================
    # ✅ [PATCH 1] _get_current_location as proper class method (indent fixed)
    # =========================
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

        cmds = []
        cmds.append(self.client.takeoffAsync(vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.takeoffAsync(vehicle_name=agent))
        cmds.append(self.client.takeoffAsync(vehicle_name=self.dynamic_name))

        for c in cmds:
            c.join()

        self.start_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.start_location[agent] = self.client.simGetObjectPose(agent)
        self.start_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

        cmds = []
        cmds.append(self.client.moveToPositionAsync(
            x=0.0, y=0.0, z=self.fixed_z,
            velocity=5.0, vehicle_name="Drone1"
        ))

        for agent in self.possible_agents:
            cmds.append(self.client.moveToPositionAsync(
                x=0.0, y=0.0, z=self.fixed_z,
                velocity=5.0, vehicle_name=agent
            ))

        cmds.append(self.client.moveToPositionAsync(
            x=0.0, y=0.0, z=self.fixed_z,
            velocity=5.0, vehicle_name=self.dynamic_name
        ))

        for c in cmds:
            c.join()

        self._hover("Drone1")
        for agent in self.possible_agents:
            self._hover(agent)
        self._hover(self.dynamic_name)

        time.sleep(1)

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
        is_success = 1 if status == "SUCCESS_DISTANCE_AGENT_DYNAMIC" else 0
        is_leader_hit = 1 if "LEADER" in status else 0
        is_ally_collision = 1 if "AGENT_AGENT" in status else 0
        is_obj_collision = 1 if "FAR_CUTOFF" in status else 0

        self.stats_history["win"].append(is_success)
        self.stats_history["coll_leader"].append(is_leader_hit)
        self.stats_history["coll_drone"].append(is_ally_collision)
        self.stats_history["coll_obj"].append(is_obj_collision)

        def get_rate(key):
            if len(self.stats_history[key]) > 0:
                return sum(self.stats_history[key]) / len(self.stats_history[key])
            return 0.0

        win_rate = get_rate("win")
        coll_rate_leader = get_rate("coll_leader")
        coll_rate_drone = get_rate("coll_drone")
        coll_rate_obj = get_rate("coll_obj")

        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        for agent in self.possible_agents:
            _obs_list.append(self._get_obs(agent))
            _rewards_list.append(reward)
            _terminations_list.append(True)
            _infos_list.append({
                agent: {
                    "final_status": status,
                    "reward": reward,
                    "episode_success": is_success,
                    "episode_leader_hit": is_leader_hit,
                    "episode_ally_collision": is_ally_collision,
                    "win_rate": win_rate,
                    "coll_rate_leader": coll_rate_leader,
                    "coll_rate_drone": coll_rate_drone,
                    "coll_rate_obj": coll_rate_obj,
                    "difficulty_stage": self.difficulty_stage,
                    "cur_episode_steps": self.step_count
                }
            })

        print(f"[{self.episode_count} Ep] Stage: {self.difficulty_stage} | WinRate: {win_rate:.2f} | Status: {status}")
        return _obs_list, _rewards_list, _terminations_list, _infos_list

    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self):
        self._obs_step_timer += 1

        if len(self.stats_history["win"]) >= 20:
            current_win_rate = sum(self.stats_history["win"]) / len(self.stats_history["win"])
            if current_win_rate >= 0.7 and self.difficulty_stage < 3:
                self.difficulty_stage += 1
                self.stats_history["win"].clear()
                print(f"\n🚀 Level Up! Stage {self.difficulty_stage} (Win Rate: {current_win_rate:.2f}) 🚀\n")

        speeds = [0.1, 1.0, 2.0, 5.0]
        target_speed = speeds[self.difficulty_stage]

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

        if self.difficulty_stage == 0:
            radius = 20.0
        elif self.difficulty_stage == 1:
            radius = 30.0
        elif self.difficulty_stage == 2:
            radius = 40.0
        else:
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
        my_state = self.client.getMultirotorState(vehicle_name=agent)
        my_pos = np.array([
            my_state.kinematics_estimated.position.x_val,
            my_state.kinematics_estimated.position.y_val
        ], dtype=np.float32)

        v_world_x = my_state.kinematics_estimated.linear_velocity.x_val
        v_world_y = my_state.kinematics_estimated.linear_velocity.y_val

        q = my_state.kinematics_estimated.orientation
        _, _, my_yaw = airsim.to_eularian_angles(q)

        sin_yaw = math.sin(my_yaw)
        cos_yaw = math.cos(my_yaw)

        v_x_body = v_world_x * math.cos(my_yaw) + v_world_y * math.sin(my_yaw)
        v_y_body = -v_world_x * math.sin(my_yaw) + v_world_y * math.cos(my_yaw)

        yaw_rate = my_state.kinematics_estimated.angular_velocity.z_val

        norm_vx_body = np.clip(v_x_body / self.MAX_SPEED, -1.0, 1.0)
        norm_vy_body = np.clip(v_y_body / self.MAX_SPEED, -1.0, 1.0)
        norm_yaw_rate = np.clip(yaw_rate / 2.0, -1.0, 1.0)

        target_state = self.client.getMultirotorState(vehicle_name=self.dynamic_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val
        ], dtype=np.float32)
        target_vel = np.array([
            target_state.kinematics_estimated.linear_velocity.x_val,
            target_state.kinematics_estimated.linear_velocity.y_val
        ], dtype=np.float32)

        my_vel_global = np.array([v_world_x, v_world_y], dtype=np.float32)

        R_vec = target_pos - my_pos
        V_vec = target_vel - my_vel_global

        dist = float(np.linalg.norm(R_vec))
        epsilon = 1e-6

        closing_speed = -float(np.dot(R_vec, V_vec)) / (dist + epsilon)
        cross_prod = float(R_vec[0] * V_vec[1] - R_vec[1] * V_vec[0])
        los_rate = cross_prod / (dist**2 + epsilon)

        # raw clip
        closing_speed = float(np.clip(closing_speed, -30.0, 30.0))
        los_rate = float(np.clip(los_rate, -10.0, 10.0))

        # =========================
        # ✅ [PATCH 3] normalize closing/los -> /30, /10 to [-1,1]
        # =========================
        closing_norm = float(np.clip(closing_speed / 30.0, -1.0, 1.0))
        los_norm = float(np.clip(los_rate / 10.0, -1.0, 1.0))

        _leader_feats = []
        _ally_feats = []

        _leader_feats = [self._angle_and_distance(agent, "Drone1")]

        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            _ally_feats.append(self._angle_and_distance(agent, other))

        base_dynamic_feat = self._angle_and_distance(agent, self.dynamic_name)
        _dynamic_feats = [
            base_dynamic_feat[0],   # bearing (normalized)
            base_dynamic_feat[1],   # distance
            closing_norm,           # normalized closing
            los_norm                # normalized los
        ]

        obs = np.concatenate([
            np.array(_leader_feats, dtype=np.float32).flatten(),
            np.array(_ally_feats, dtype=np.float32).flatten(),
            np.array(_dynamic_feats, dtype=np.float32).flatten(),
            np.array([norm_vx_body, norm_vy_body, norm_yaw_rate, sin_yaw, cos_yaw], dtype=np.float32)
        ]).astype(np.float32)

        return obs

    def _do_action(self, actions):
        actions = np.clip(actions, -1, 1)

        # =========================
        # ✅ [PATCH 4] dt unify -> self.dt
        # =========================
        dt = self.dt

        for i, agent in enumerate(self.possible_agents):
            a = actions[i]

            yaw_rate = float(a[0] * self.MAX_YAW)

            v_forward = (float(a[1]) + 1.0) / 2.0 * self.MAX_SPEED
            v_lateral = float(a[2]) * self.MAX_SPEED

            sp = math.hypot(v_forward, v_lateral)
            if sp > self.MAX_SPEED:
                s = self.MAX_SPEED / (sp + 1e-6)
                v_forward *= s
                v_lateral *= s

            st = self.client.getMultirotorState(vehicle_name=agent)
            q = st.kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(q)

            vx = v_forward * math.cos(yaw) - v_lateral * math.sin(yaw)
            vy = v_forward * math.sin(yaw) + v_lateral * math.cos(yaw)

            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate))

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

        # heading_state 초기화 (현재 코드에선 실질적으로 안 씀, 남겨둠)
        self.heading_state = {a: 0.0 for a in self.possible_agents}

        self.client.reset()
        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        self._reset_obstacle_logic()
        self._get_current_location()

        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}

        self._last_pose.clear()
        self._last_time.clear()

        # =========================
        # ✅ [PATCH 5] reset에서도 action buffer (3,)
        # =========================
        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}

        self.leader_stop = False

        obs_list = [self._get_obs(a) for a in self.agents]
        print("reset.")
        return obs_list

    def step(self, actions):
        self.step_count += 1

        if self.step_count >= 300:
            print(f"⏳[시간 초과] 스텝 {self.step_count} 도달! → 리더 피격 처리(실패)로 강제 종료")
            # 리더 피격 보상(REWARD_LEADER_HIT, -250.0)을 주어 패널티 부여
            return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_TIMEOUT_LEADER_HIT")
        
        per_agent_obs, per_agent_results, per_agent_infos = [], [], []

        self._do_action(actions)
        self._update_leader_movement()
        self._update_dynamic_obstacle()
        self._get_current_location()

        for agent in self.possible_agents:
            other_agents = [a for a in self.possible_agents if a != agent]

            _distance_leader = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location["Drone1"].position.x_val,
                self.current_location[agent].position.y_val - self.current_location["Drone1"].position.y_val,
                self.current_location[agent].position.z_val - self.current_location["Drone1"].position.z_val
            ])

            _distance_dynamic = np.linalg.norm([
                self.current_location[agent].position.x_val - self.current_location[self.dynamic_name].position.x_val,
                self.current_location[agent].position.y_val - self.current_location[self.dynamic_name].position.y_val,
                self.current_location[agent].position.z_val - self.current_location[self.dynamic_name].position.z_val
            ])

            if _distance_leader > self.far_cutoff:
                print(
                    f"[이탈] {agent}가 리더와의 거리({_distance_leader:.2f}m)로, "
                    f"이탈 임계값({self.far_cutoff}m) 초과! → 전체 실패(경계 이탈)"
                )
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_AGENT_FAR_CUTOFF")

            hit, d = self._check_distance_collision(agent, "Drone1", threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"⚠️💔[거리-충돌] {agent} ↔ Drone1  (d={d:.2f}) → 전체 실패")
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_LEADER")

            for other in other_agents:
                hit, d = self._check_distance_collision(agent, other, threshold=self.COLLISION_THRESHOLD)
                if hit:
                    print(f"💥🤖[거리-충돌] {agent} ↔ {other}  (d={d:.2f}) → 전체 실패")
                    return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_AGENT")

            collisionInfo = self.client.simGetCollisionInfo("Drone1")
            if collisionInfo.has_collided and collisionInfo.object_name == self.dynamic_name:
                print(f"💥[충돌] 유인기가 {collisionInfo.object_name}와 충돌로 → 전체 실패")
                return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_LEADER_AND_DYNAMIC_OBSTACLE_COLLISION")

            hit, d = self._check_distance_collision(agent, self.dynamic_name, threshold=self.COLLISION_THRESHOLD)
            if hit:
                print(f"🎯🔥[거리-충돌] {agent} ↔ {self.dynamic_name}  (d={d:.2f}) → 요격 성공")
                return self._end_episode(self.REWARD_HIT_ENEMY, "SUCCESS_DISTANCE_AGENT_DYNAMIC")

            per_agent_obs.append(self._get_obs(agent))

            _reward, _ = self._compute_reward(agent)
            per_agent_results.append(_reward)
            per_agent_infos.append([f"reward: {_reward}"])

        termination_list = [False for _ in self.possible_agents]
        rewards_list = self._get_rewards(per_agent_results)
        obs_list = per_agent_obs
        infos_list = per_agent_infos

        return obs_list, rewards_list, termination_list, infos_list
