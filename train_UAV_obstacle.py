#!/usr/bin/env python
import os
from pathlib import Path
import sys
import numpy as np
import torch
import wandb

try:
    import setproctitle
    _HAS_SETPROC = True
except Exception:
    _HAS_SETPROC = False

# --- 프로젝트 파일 임포트 ---
from onpolicy.config_UAV import get_config
from onpolicy.envs.make_env_PN_Lidar import AirSimMultiDroneEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import gym


# ===== 멀티에이전트 → MAPPO 배치 텐서로 변환하는 래퍼 =====
class AirSimMAPPOGymWrapper(gym.Env):
    """
    내부에 AirSimMultiDroneEnv(에이전트 리스트 기반)를 들고,
    reset/step에서 (num_agents, obs_dim) 배치 텐서를 반환하도록 맞춰줍니다.
    """
    metadata = {"render_modes": []}

    def __init__(self, base_env: AirSimMultiDroneEnv):
        super().__init__()
        self.base = base_env
        self.agents = self.base.possible_agents
        self.num_agents = len(self.agents)

        # observation/action space를 '공통 스페이스'로 노출
        # (onpolicy 러너는 공유 폼을 기대)
        self.single_observation_space = self.base.observation_spaces[self.agents[0]]
        self.single_action_space = self.base.action_spaces[self.agents[0]]

        self._obs_spaces = [self.single_observation_space for _ in range(self.num_agents)]
        self._act_spaces = [self.single_action_space      for _ in range(self.num_agents)]

         # self._batched_observation_space = gym.spaces.Box(...)       

        '''
        # 벡터라이즈된 형태처럼 보이게 기본 속성 제공
        self.observation_space = gym.spaces.Box(
            low=np.repeat(self.single_observation_space.low[None, :], self.num_agents, axis=0),
            high=np.repeat(self.single_observation_space.high[None, :], self.num_agents, axis=0),
            shape=(self.num_agents, *self.single_observation_space.shape),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.repeat(self.single_action_space.low[None, :], self.num_agents, axis=0),
            high=np.repeat(self.single_action_space.high[None, :], self.num_agents, axis=0),
            shape=(self.num_agents, *self.single_action_space.shape),
            dtype=np.float32,
        )
        '''
    @property
    def observation_space(self):
        return self._obs_spaces
    @property
    def action_space(self):
        return self._act_spaces
    @property
    def share_observation_space(self):
        # base가 가진 Box(share_obs_dim,)를 에이전트 수만큼 리스트로
        return [self.base.share_observation_spaces for _ in self.base.possible_agents]
  

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.base.seed(seed)
        obs_list = self.base.reset(seed=seed, options=options)
        obs = np.asarray(obs_list, dtype=np.float32)   # (num_agents, obs_dim)
        return obs



    def step(self, actions):
        # actions: (num_agents, act_dim) 또는 list
        if isinstance(actions, np.ndarray):
            act_list = actions.tolist()
        else:
            act_list = actions

        obs_list, rewards_list, dones_list, infos_list = self.base.step(act_list)

        obs     = np.asarray(obs_list, dtype=np.float32)           # (num_agents, obs_dim)

        # ✅ 보상과 done을 (num_agents, 1) 형태로 맞춰줌
        rewards = np.asarray(rewards_list, dtype=np.float32)[:, None]  # (num_agents, 1)
        dones   = np.asarray(dones_list,  dtype=np.bool_)[:,   None]   # (num_agents, 1)

        infos   = {"infos": infos_list}
        return obs, rewards, dones, infos



    # onpolicy DummyVecEnv/SubprocVecEnv이 호출하는 인터페이스 보강
    def seed(self, seed):
        self.base.seed(seed)
        return [seed]

    def close(self):
        try:
            # 필요 시 AirSim 정리 루틴을 넣을 수 있음
            pass
        except Exception:
            pass


# ====== AirSim은 프로세스 간 공유가 어려움 → 항상 DummyVecEnv 권장 ======
def _build_single_env(all_args, rank_seed):
    base_env = AirSimMultiDroneEnv(
        ip_address=all_args.ip_address,
        follower_names=[f"Follower{i}" for i in range(all_args.num_agents)],
        step_length=all_args.step_length,
        #leader_velocity=all_args.leader_velocity,
        optimal_distance=all_args.optimal_distance
    )
    base_env.seed(rank_seed)
    return AirSimMAPPOGymWrapper(base_env)


def make_train_env(all_args):
    # AirSim은 SubprocVecEnv(멀티프로세스)에서 자주 크래시 → 강제로 1 스레드
    if all_args.n_rollout_threads != 1:
        print("[WARN] AirSim 환경은 멀티프로세싱과 궁합이 안 좋아서 n_rollout_threads=1로 강제합니다.")
        all_args.n_rollout_threads = 1

    def get_env_fn(rank):
        def init_env():
            return _build_single_env(all_args, all_args.seed + rank * 1000)
        return init_env

    return DummyVecEnv([get_env_fn(0)])


def make_eval_env(all_args):
    if not all_args.use_eval:
        return None

    if all_args.n_eval_rollout_threads != 1:
        print("[WARN] AirSim 평가 환경도 n_eval_rollout_threads=1로 강제합니다.")
        all_args.n_eval_rollout_threads = 1

    def get_env_fn(rank):
        def init_env():
            return _build_single_env(all_args, all_args.seed * 50000 + rank * 10000)
        return init_env

    return DummyVecEnv([get_env_fn(0)])


def main(args):
    parser = get_config()
    all_args = parser.parse_args(args)

    # 알고리즘 선택
    if all_args.algorithm_name == "rmappo":
        print("알고리즘: rmappo → use_recurrent_policy=True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("알고리즘: mappo → use_recurrent_policy=False, use_naive_recurrent_policy=False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError("rmappo 또는 mappo 알고리즘만 지원합니다.")

    # CUDA 설정
    if getattr(all_args, "cuda", False) and torch.cuda.is_available():
        print("GPU를 사용합니다.")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if getattr(all_args, "cuda_deterministic", False):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("CPU를 사용합니다.")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 결과 저장 경로
    run_dir = Path(r"C:\Users\rlatn\Desktop\RL_Project\UAV_Airsim_RL_Project\on-policy\final\PN_LIDAR")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    run = None
    if all_args.use_wandb:
        run = wandb.init(
            config=vars(all_args),
            project=all_args.env_name,
            name="-".join([f"Lidar적용_dt0.1+rollout2048"]),
            dir=str(run_dir),
            job_type="training",
            reinit=True
        )

    # 프로세스 이름
    if _HAS_SETPROC:
        try:
            setproctitle.setproctitle("-".join([all_args.env_name, all_args.algorithm_name, all_args.experiment_name]))
        except Exception:
            pass

    # 시드
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Env 초기화 (항상 DummyVecEnv)
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # Runner 임포트
    if all_args.share_policy:
        from onpolicy.runner.shared.uav_runner_obstacle import Runner
    else:
        from onpolicy.runner.separated.base_runner import BaseRunner as Runner

    # 안전 실행
    try:
        runner = Runner(config)
        runner.run()
    finally:
        try:
            envs.close()
        except Exception:
            pass
        if all_args.use_eval and eval_envs is not None and eval_envs is not envs:
            try:
                eval_envs.close()
            except Exception:
                pass
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main(sys.argv[1:])
