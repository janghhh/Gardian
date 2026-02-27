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
from onpolicy.envs.make_env_PN import AirSimMultiDroneEnv
from onpolicy.envs.env_wrappers import DummyVecEnv
import gym


# ===== 멀티에이전트 → MAPPO 배치 텐서로 변환하는 래퍼 =====
class AirSimMAPPOGymWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_env: AirSimMultiDroneEnv):
        super().__init__()
        self.base = base_env
        self.agents = self.base.possible_agents
        self.num_agents = len(self.agents)

        self.single_observation_space = self.base.observation_spaces[self.agents[0]]
        self.single_action_space = self.base.action_spaces[self.agents[0]]
        self._obs_spaces = [self.single_observation_space for _ in range(self.num_agents)]
        self._act_spaces = [self.single_action_space for _ in range(self.num_agents)]

    @property
    def observation_space(self):
        return self._obs_spaces

    @property
    def action_space(self):
        return self._act_spaces

    @property
    def share_observation_space(self):
        return [self.base.share_observation_spaces for _ in self.base.possible_agents]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.base.seed(seed)
        obs_list = self.base.reset(seed=seed, options=options)
        obs = np.asarray(obs_list, dtype=np.float32)
        return obs

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            act_list = actions.tolist()
        else:
            act_list = actions

        obs_list, rewards_list, dones_list, infos_list = self.base.step(act_list)
        obs = np.asarray(obs_list, dtype=np.float32)
        rewards = np.asarray(rewards_list, dtype=np.float32)[:, None]
        dones = np.asarray(dones_list, dtype=np.bool_)[:, None]
        infos = {"infos": infos_list}
        return obs, rewards, dones, infos

    def seed(self, seed):
        self.base.seed(seed)
        return [seed]

    def close(self):
        try:
            pass
        except Exception:
            pass


# ===== AirSim DummyVecEnv =====
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
    run_dir = Path(r"C:\Users\rlatn\Desktop\RL_Project\UAV_Airsim_RL_Project\on-policy\final\PN_2action_dxdy_level2")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    run = None
    if all_args.use_wandb:
        run = wandb.init(
            config=vars(all_args),
            project=all_args.env_name,
            name="-".join([f"[난이도 2]PN+액션 요우 삭제+dx,dy관측"]),
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

    # ---------------------------
    # ✅ 기존 학습된 모델 불러오기
    # ---------------------------
    actor_path = r"C:\Users\rlatn\Desktop\RL_Project\UAV_Airsim_RL_Project\on-policy\final\PN_2action_dxdy_level1\wandb\success_model\files\actor.pt"
    critic_path = r"C:\Users\rlatn\Desktop\RL_Project\UAV_Airsim_RL_Project\on-policy\final\PN_2action_dxdy_level1\wandb\success_model\files\critic.pt"

    runner = Runner(config)

    if os.path.exists(actor_path) and os.path.exists(critic_path):
        print(f"✅ 기존 학습된 모델 불러오기 완료\n  Actor: {actor_path}\n  Critic: {critic_path}")
        actor_state = torch.load(actor_path, map_location=runner.device)
        critic_state = torch.load(critic_path, map_location=runner.device)
        runner.policy.actor.load_state_dict(actor_state)
        runner.policy.critic.load_state_dict(critic_state)
        print("🔥 기존 정책 기반으로 이어서 학습을 시작합니다.")
    else:
        print("⚠️ 저장된 정책이 없어 새로 학습을 시작합니다.")

    # ---------------------------
    # 학습 시작
    # ---------------------------
    try:
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
