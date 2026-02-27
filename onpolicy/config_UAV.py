import argparse

def get_config():
    """
    Airsim 드론 편대비행 프로젝트를 위한 설정값 파서.
    rMAPPO/MAPPO 알고리즘과 관련된 모든 파라미터를 포함한 최종 버전입니다.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # ==================================================================================================
    # 필수 설정 (Experiment & General)
    # ==================================================================================================
    parser.add_argument("--algorithm_name", type=str,
                        default='rmappo', choices=["rmappo", "mappo"],
                        help="사용할 알고리즘 선택")
    parser.add_argument("--experiment_name", type=str, default="drone_formation_v1", 
                        help="실험 구분을 위한 이름")
    parser.add_argument("--seed", type=int, default=1, 
                        help="Numpy/Torch 랜덤 시드")
    parser.add_argument("--n_training_threads", type=int, default=1, 
                        help="학습에 사용할 CPU 스레드 수")
    parser.add_argument("--n_rollout_threads", type=int, default=1,
                        help="데이터 수집에 사용할 병렬 환경 수")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="평가에 사용할 병렬 환경 수")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="렌더링에 사용할 병렬 환경 수")
    parser.add_argument("--num_env_steps", type=int, default=3000000,
                        help="총 학습 스텝 수")
    parser.add_argument("--cuda", action='store_false', default=True, 
                        help="GPU 사용 여부")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, 
                        help="CUDA 연산의 결정론적 동작 여부")

    # ==================================================================================================
    # Airsim 환경 전용 설정 (Environment Specific)
    # ==================================================================================================
    parser.add_argument("--env_name", type=str, default='UAV', 
                        help="환경 이름")
    parser.add_argument("--num_agents", type=int, default=4,
                        help="제어할 팔로워 드론의 수")
    parser.add_argument("--ip_address", type=str, default="127.0.0.1",
                        help="AirSim 서버 IP 주소")
    parser.add_argument("--step_length", type=float, default=1.0,
                        help="한 스텝 당 드론 이동 거리")
    parser.add_argument("--optimal_distance", type=float, default=10.0,
                        help="리더 드론과의 최적 거리")
    parser.add_argument("--leader_velocity", type=float, default=6.0,
                        help="리더 드론의 속도")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="전역 상태 대신 지역 관측을 합쳐서 Critic 입력으로 사용할지 여부")

    # ==================================================================================================
    # 버퍼 설정 (Replay Buffer)
    # ==================================================================================================
    parser.add_argument("--episode_length", type=int,
                        default=230, help="한 에피소드의 최대 스텝 길이")
    parser.add_argument("--rollout_length", type=int, default=2048,
                        help = "정책 업데이트 주기")

    # ==================================================================================================
    # 신경망 & 정책 설정 (Network & Policy)
    # ==================================================================================================
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    
    # Recurrent Policy 설정
    parser.add_argument("--use_recurrent_policy", action='store_false', default=True, 
                        help='순환 신경망(RNN) 정책 사용 여부')
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='단순 순환 신경망 정책 사용 여부')
    parser.add_argument("--recurrent_N", type=int, default=1, 
                        help="RNN 레이어의 수")
    parser.add_argument("--data_chunk_length", type=int, default=20,
                        help="RNN 학습에 사용할 데이터 묶음(chunk)의 길이")

    # ==================================================================================================
    # 옵티마이저 & PPO 알고리즘 하이퍼파라미터 (Optimizer & PPO)
    # ==================================================================================================
    parser.add_argument("--lr", type=float, default=5e-4, help='학습률 (learning rate)')
    parser.add_argument("--critic_lr", type=float, default=5e-4, help='Critic 학습률')
    parser.add_argument("--opti_eps", type=float, default=1e-5, help='옵티마이저 epsilon 값')
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--ppo_epoch", type=int, default=10, help='PPO 에폭 수')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='PPO 미니배치 수')
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='엔트로피 보너스 계수')
    parser.add_argument("--value_loss_coef", type=float, default=1, help='가치 손실 계수')
    parser.add_argument("--clip_param", type=float, default=0.2, help='PPO 클리핑 파라미터')
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True, 
                        help="클리핑된 가치 손실 사용 여부")
    
    parser.add_argument("--use_max_grad_norm", action='store_false', default=True, 
                        help="그래디언트 클리핑 사용 여부")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='그래디언트 클리핑 최대값')
    
    parser.add_argument("--use_gae", action='store_false', default=True, 
                        help='GAE(Generalized Advantage Estimation) 사용 여부')
    parser.add_argument("--gamma", type=float, default=0.99, help='할인율 (discount factor)')
    parser.add_argument("--gae_lambda", type=float, default=0.95, help='GAE 람다 파라미터')
    parser.add_argument("--use_proper_time_limits", action='store_true', default=False, 
                        help='에피소드 종료 시점(time limits)을 고려하여 리턴 계산 여부')
    
    parser.add_argument("--use_huber_loss", action='store_false', default=True, 
                        help="Huber 손실 함수 사용 여부")
    parser.add_argument("--huber_delta", type=float, default=10.0, 
                        help="Huber 손실 델타 값")
    
    parser.add_argument("--use_value_active_masks", action='store_false', default=True,
                        help="가치 손실 계산 시 유효한 데이터 마스크 사용 여부")
    parser.add_argument("--use_policy_active_masks", action='store_false', default=True,
                        help="정책 손실 계산 시 유효한 데이터 마스크 사용 여부")

    # ==================================================================================================
    # 실행, 로그, 저장, 평가, 렌더링 설정 (Execution, Log, Save, Eval, Render)
    # ==================================================================================================
    parser.add_argument("--use_linear_lr_decay", action='store_true', default=False, 
                        help='학습률 선형 감소 사용 여부')
    
    parser.add_argument("--save_interval", type=int, default=25, help="모델 저장 주기 (훈련 주기 기준)")
    parser.add_argument("--log_interval", type=int, default=5, help="로그 출력 주기 (훈련 주기 기준)")
    parser.add_argument("--model_dir", type=str, default=None, help="사전 학습된 모델을 불러올 경로")

    parser.add_argument("--use_eval", action='store_true', default=False, 
                        help="학습 중 주기적으로 평가를 실행할지 여부")
    parser.add_argument("--eval_interval", type=int, default=25, help="평가 실행 주기 (훈련 주기 기준)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="평가 시 실행할 에피소드 수")

    parser.add_argument("--use_render", action='store_true', default=False, help="학습 중 렌더링 여부")
    parser.add_argument("--save_gifs", action='store_true', default=False, help="결과를 gif로 저장할지 여부")

    parser.add_argument("--use_wandb", action='store_false', default=True, 
                        help="Wandb를 사용한 로그 기록 여부")

    return parser