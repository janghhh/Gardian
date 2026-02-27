import os
import torch
import numpy as np
import gym
from gym import spaces
from types import SimpleNamespace

# === 프로젝트 모듈 ===
from onpolicy.envs.make_env_PN import AirSimMultiDroneEnv
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic

# ---------- 경로 ----------
actor_model_path = r"C:\Users\rlatn\Desktop\RL_Project\UAV_Airsim_RL_Project\on-policy\final\PN_curriculum_3action_yawchange\wandb\success_model\files\actor.pt"
device = torch.device("cpu")

# ---------- Env ----------
# AirSimMultiDroneEnv 클래스는 제공된 코드에 정의되어 있습니다.
env = AirSimMultiDroneEnv(ip_address="127.0.0.1",
                          follower_names=("Follower0","Follower1","Follower2"))
obs_list = env.reset()
num_agents = len(obs_list)

# 관측/행동 space 가져오기
def _get_first_space(env, attr):
    if hasattr(env, attr):
        sp = getattr(env, attr)
        if isinstance(sp, (list, tuple)):
            return sp[0]
        return sp
    return None

obs_space = _get_first_space(env, "observation_space")
act_space = _get_first_space(env, "action_space")

# 없다면 첫 관측으로 유추 (제공된 환경에서는 이미 정의되어 있음)
if obs_space is None:
    obs_dim = int(np.asarray(obs_list[0]).size)
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

if act_space is None:
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

args = SimpleNamespace(
    hidden_size=64,
    gain=0.01,
    use_orthogonal=True,
    use_policy_active_masks=False,
    use_naive_recurrent_policy=False, 
    use_recurrent_policy=True,         
    recurrent_N=1,
    algorithm_name="rmappo",
    use_popart=False,
    
    # --- ★ 필수 MLPBase/CNNBase 필드 (오류 해결 완료) ★ ---
    use_feature_normalization=True,
    use_cnn_feature=False,           
    stacked_frames=1,
    activation_id=1,
    use_ReLU=True,                   
    layer_N=1,                       
)

# ---------- Actor 인스턴스 & state_dict 로드 ----------
# 이제 args에 use_feature_normalization이 포함되어 오류가 해결됩니다.
actor = R_Actor(args, obs_space, act_space, device=device)
state = torch.load(actor_model_path, map_location=device)
state_dict = state.get("model", state) if isinstance(state, dict) else state
actor.load_state_dict(state_dict, strict=True)
actor.eval()

# ---------- 평가 루프 (변경 없음) ----------
def evaluate_policy(env, actor, num_episodes=5, deterministic=True):
    total_rewards = []
    use_rnn = args.use_naive_recurrent_policy or args.use_recurrent_policy
    hidden_size = args.hidden_size

    for ep in range(num_episodes):
        obs_list = env.reset()
        done_list = [False] * len(obs_list)
        ep_ret = [0.0] * len(obs_list)

        if use_rnn:
            rnn_states = torch.zeros((len(obs_list), args.recurrent_N, hidden_size), dtype=torch.float32, device=device)
        else:
            rnn_states = torch.zeros((len(obs_list), args.recurrent_N, hidden_size), dtype=torch.float32, device=device)

        masks = torch.ones((len(obs_list), 1), dtype=torch.float32, device=device)

        while not all(done_list):
            actions = []
            
            # 모든 에이전트 관측을 한 번에 처리하도록 개선 가능하지만, 현재 루프 유지
            for i, obs in enumerate(obs_list):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                rs = rnn_states[i:i+1]
                mk = masks[i:i+1]

                with torch.no_grad():
                    act_t, _, rs_out = actor(obs_tensor, rs, mk, available_actions=None, deterministic=deterministic)
                
                act = act_t.squeeze(0).cpu().numpy()
                actions.append(act)

                if use_rnn:
                    rnn_states[i:i+1] = rs_out

            next_obs_list, rewards_list, done_list, info = env.step(actions)

            if use_rnn:
                for i, d in enumerate(done_list):
                    masks[i:i+1] = torch.tensor([[0.0 if d else 1.0]], dtype=torch.float32, device=device)

            for i, r in enumerate(rewards_list):
                ep_ret[i] += float(r)

            obs_list = next_obs_list

        mean_ep = float(np.mean(ep_ret))
        total_rewards.append(mean_ep)
        print(f"[Episode {ep+1}] mean reward: {mean_ep:.2f}")

    print(f"Average over {num_episodes} eps: {float(np.mean(total_rewards)):.2f}")

# 실행
print("Policy evaluation started...")
evaluate_policy(env, actor, num_episodes=100, deterministic=True)
print("Policy evaluation finished.")