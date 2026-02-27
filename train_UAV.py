#!/usr/bin/env python
import os
from pathlib import Path
import sys
import socket
import numpy as np
import setproctitle
import torch
import wandb

# --- 프로젝트 파일 임포트 ---
from onpolicy.config_UAV import get_config
from onpolicy.envs.make_env_PN import AirSimMultiDroneEnv 
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = AirSimMultiDroneEnv(
                ip_address=all_args.ip_address,
                follower_names=[f"Follower{i}" for i in range(all_args.num_agents)],
                step_length=all_args.step_length,
                leader_velocity=all_args.leader_velocity,
                optimal_distance=all_args.optimal_distance
            )
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = AirSimMultiDroneEnv(
                ip_address=all_args.ip_address,
                follower_names=[f"Follower{i}" for i in range(all_args.num_agents)],
                step_length=all_args.step_length,
                leader_velocity=all_args.leader_velocity,
                optimal_distance=all_args.optimal_distance
            )
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def main(args):
    parser = get_config()
    all_args = parser.parse_args(args)

    # 알고리즘 선택 로직
    if all_args.algorithm_name == "rmappo":
        print("알고리즘: rmappo, use_recurrent_policy=True로 설정합니다.")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("알고리즘: mappo, use_recurrent_policy=False, use_naive_recurrent_policy=False로 설정합니다.")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError("rmappo 또는 mappo 알고리즘만 지원합니다.")

    # CUDA 설정
    if all_args.cuda and torch.cuda.is_available():
        print("GPU를 사용합니다.")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("CPU를 사용합니다.")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 결과 저장 경로 설정
    run_dir = Path("./results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Wandb 설정
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         name="-".join([f"PN+적군속도0부터커리큘럼학습"]),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    
    # 프로세스 이름 설정
    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]))
    
    # 시드 고정
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Env 초기화
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
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
        from onpolicy.runner.shared.uav_runner import Runner
    else:
        from onpolicy.runner.separated.base_runner import BaseRunner as Runner

    runner = Runner(config)
    runner.run()
    
    # 후처리
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()

if __name__ == "__main__":
    main(sys.argv[1:])