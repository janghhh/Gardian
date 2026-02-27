import wandb
import os
import numpy as np
import torch
import time
import gc
from collections import defaultdict
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer_uav import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], self.num_agents, device = self.device)
        else:
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device = self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        else:
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        try:
            self.rollout_length = self.all_args.rollout_length
        except AttributeError:
            # --rollout_length가 없으면, 기존 방식(episode_length)으로 작동
            print("Warning: --rollout_length not specified. Using episode_length as rollout_length.")
            self.rollout_length = self.all_args.episode_length

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

        # ---------------------------------------------------------------------
        # ★ 누적 보상(Cumulative Reward) 로깅을 위한 변수 초기화
        # ---------------------------------------------------------------------
        # 1. (병렬) 스레드별로 현재 에피소드의 보상을 누적하는 변수
        #    (n_threads, 1) 모양으로 초기화. (환경이 팀 보상을 반환하므로 1개만 추적)
        self.current_episode_rewards = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
        
        # 2. 로그 간격 동안 완료된 에피소드들의 최종 점수를 저장하는 리스트
        self.finished_episode_scores = []
        # ---------------------------------------------------------------------


    def run(self):
        """
        메인 루프. [롤아웃 버퍼(rollout_length) 단위로 업데이트하도록 수정됨]
        """
        self.warmup()

        start = time.time()

        # ★ 1. 메인 루프를 '총 업데이트 횟수' 기준으로 변경
        #    (기존: episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads)
        num_updates = int(self.num_env_steps) // self.rollout_length // self.n_rollout_threads
        
        # (기존 L.131) 'episode' -> 'update'
        for update in range(num_updates):
            if self.use_linear_lr_decay:
                # (기존 L.132) 'episode' -> 'update'
                self.trainer.policy.lr_decay(update, num_updates)

            # (기존 L.135) 1 롤아웃(예: 8192 스텝) 동안의 모든 보상을 누적하기 위해 초기화
            ep_rewards_detailed = defaultdict(list)
            ep_success_flags = []         # episode_success (0 or 1)
            ep_leader_hit_flags = []      # episode_leader_hit (0 or 1)
            ep_ally_collision_flags = []  # episode_ally_collision (0 or 1)
            # ★ 2. 데이터 수집 루프를 'rollout_length' (예: 8192 스텝)로 변경
            # (기존 L.137) 'self.episode_length' -> 'self.rollout_length'
            for step in range(self.rollout_length):
                
                # (L.138) 행동 결정 (변경 없음)
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                
                # (L.139) 환경 실행 (변경 없음)
                obs, rewards, dones, infos = self.envs.step(actions)

                # ---------------------------------------------------------------------
                # ★ 누적 보상(Cumulative Reward) 계산
                # ---------------------------------------------------------------------
                # 1. 현재 스텝의 보상(팀 보상)을 스레드별 누적 변수에 더함
                #    (rewards shape: (n_threads, n_agents, 1), 팀 보상이므로 0번 에이전트만 사용)
                self.current_episode_rewards += rewards[:, 0, :]

                # 2. 완료된 스레드(에피소드)가 있는지 확인
                #    (dones shape: (n_threads, n_agents, 1))
                for i in range(self.n_rollout_threads):
                    # (팀 전체가 동시에 done되므로 에이전트 0번만 체크)
                    if dones[i, 0, 0]: 
                        # 2-1. 완료된 에피소드의 최종 누적 점수를 리스트에 추가
                        self.finished_episode_scores.append(self.current_episode_rewards[i, 0])
                        # 2-2. 해당 스레드의 누적 점수를 0으로 리셋
                        self.current_episode_rewards[i, 0] = 0.0

                        info_i = infos[i]   # i번째 환경의 info 딕셔너리

                        # env 구현에 따라 두 가지 케이스를 모두 커버:
                        # 1) info = {"episode_success": 1, "episode_leader_hit": 0, ...}
                        # 2) info = {"Follower0": {..., "episode_success": 1, ...}, ...}
                        success_flag = None
                        leader_hit_flag = 0
                        ally_collision_flag = 0

                        # 케이스 1: info = {"episode_success": 1, ...}
                        if isinstance(info_i, dict) and "episode_success" in info_i:
                            success_flag = info_i.get("episode_success", 0)
                            leader_hit_flag = info_i.get("episode_leader_hit", 0)
                            ally_collision_flag = info_i.get("episode_ally_collision", 0)

                        else:
                            # 케이스 2: info = {"Follower0": {...}, ...}
                            if isinstance(info_i, dict):
                                for v in info_i.values():
                                    if isinstance(v, dict) and "episode_success" in v:
                                        success_flag = v.get("episode_success", 0)
                                        leader_hit_flag = v.get("episode_leader_hit", 0)
                                        ally_collision_flag = v.get("episode_ally_collision", 0)
                                        break

                            # 케이스 3: wrapper가 싸준 형태
                            # info = {"infos": [ {"Follower0": {...}}, {"Follower1": {...}}, ... ]}
                            if success_flag is None and isinstance(info_i, dict) and "infos" in info_i:
                                for per_agent in info_i["infos"]:
                                    if not isinstance(per_agent, dict):
                                        continue
                                    for inner in per_agent.values():
                                        if isinstance(inner, dict) and "episode_success" in inner:
                                            success_flag = inner.get("episode_success", 0)
                                            leader_hit_flag = inner.get("episode_leader_hit", 0)
                                            ally_collision_flag = inner.get("episode_ally_collision", 0)
                                            break
                                    if success_flag is not None:
                                        break

                        # 플래그가 제대로 발견됐으면 리스트에 저장
                        if success_flag is not None:
                            ep_success_flags.append(success_flag)
                            ep_leader_hit_flags.append(leader_hit_flag)
                            ep_ally_collision_flags.append(ally_collision_flag)
                # ---------------------------------------------------------------------
                
                # (L.141~L.145) 로깅을 위한 보상 정보 추출 (변경 없음)
                # 8192 스텝 동안 발생하는 모든 보상 내역을 누적
                for info in infos:
                    if "rewards" in info:
                        for k, v in info["rewards"].items():
                            ep_rewards_detailed[k].append(v)

                # (L.146) 버퍼에 데이터 삽입 (변경 없음)
                # 이 insert 함수가 dones=True일 때 masks=0을 기록하여
                # 8192 스텝 내의 여러 에피소드를 구분해 줍니다.
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            # ★ 3. 8192 스텝 수집이 *끝난 뒤* 업데이트를 1회만 호출!
            # (기존 L.147, L.148)
            self.compute()
            train_infos = self.train()  

            # ★ 4. 로깅 및 저장 시점 변경 (update 기준)
            # (기존 L.150) 'episode' -> 'update', 'episode_length' -> 'rollout_length'
            total_num_steps = (update + 1) * self.rollout_length * self.n_rollout_threads
            
            # (기존 L.152) 'episode' -> 'update'
            if (update % self.log_interval == 0):
                end = time.time()
                print(f"\n Env-{self.env_name} ... FPS {int(total_num_steps / (end - start))}.")
                
                # -----------------------------------------------------------------
                # ★ 로깅 블록 수정
                # -----------------------------------------------------------------
                
                # 지표 1: 'step_average_reward' (버퍼 내 모든 스텝의 평균 보상, 3.0 수렴)
                total_ep_reward = np.mean(self.buffer.rewards[self.buffer.rewards != 0])
                train_infos['step_average_reward'] = total_ep_reward
                
                # 지표 2: 'cumulative_episode_reward' (완료된 에피소드의 평균 총점, 우상향)
                if len(self.finished_episode_scores) > 0:
                    real_ep_reward = np.mean(self.finished_episode_scores)
                    train_infos['episode_average_reward'] = real_ep_reward # <-- W&B에 우상향 그래프
                    self.finished_episode_scores.clear() # 리스트 비우기
                
                # -----------------------------------------------------------------

                env_infos = {}
                for k, v in ep_rewards_detailed.items():
                    env_infos[f"reward/{k}"] = v # wandb에서 그룹으로 묶어보기 위해 이름 변경

                if len(ep_success_flags) > 0:
                    env_infos["episode/success"] = ep_success_flags
                    env_infos["episode/leader_hit"] = ep_leader_hit_flags
                    env_infos["episode/ally_collision"] = ep_ally_collision_flags
                
                # train_infos에 담긴 모든 지표(loss, step_avg, cumulative)를 한 번에 로깅
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # ★ (수정) L.194~L.201에 있던 중복/오류 코드를 위로 통합했으므로 삭제함.

            # (기존 L.174) 'episode' -> 'update'
            if (self.use_eval and update % self.eval_interval == 0):
                self.eval(total_num_steps)

            # (기존 L.178) 'episode' -> 'update', 'episodes' -> 'num_updates'
            if (update % self.save_interval == 0 or update == num_updates - 1):
                self.save()

    def warmup(self):
        """훈련 시작 전, 버퍼에 초기 관측값을 채워넣음."""
        # 초기 obs 얻기
        obs = self.envs.reset()
        
        # share_obs 처리
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        # 버퍼의 첫 스텝(0번 인덱스)에 초기값 삽입
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """정책을 이용해 행동을 결정하고, 다음 RNN 상태 등을 반환."""
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic, preds \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]))
        # 반환값을 다시 thread 수에 맞게 분리
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        predictions = np.array(np.split(_t2n(preds), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, predictions

    def insert(self, data):
        """환경으로부터 받은 데이터를 버퍼에 삽입."""
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # rewards 모양 정리:
        # - 우리 env 래퍼에서 이미 (n_threads, n_agents, 1) 로 들어오는 게 정상.
        # - 혹시 (n_threads, n_agents)로 올 땐 마지막 축을 하나 붙여 준다.
        if rewards.ndim == 2:
            rewards = rewards[..., None]   # -> (n_threads, n_agents, 1)

        # dones 마스크를 (n_threads, n_agents)로 압축해서 쓰면
        # 순환/비순환 모두에서 인덱싱이 단순하고 안전함.
        done_mask = np.squeeze(dones, axis=-1).astype(bool)   # (n_threads, n_agents)

        # --- RNN state reset (모양 가정 금지: 타깃 selection 모양 그대로 0으로 채우기) ---
        sel = rnn_states[done_mask]            # shape: (k, ..., hidden)
        if sel.size > 0:
            rnn_states[done_mask] = 0.0        # zeros_like(sel)와 동일 효과

        selv = rnn_states_critic[done_mask]
        if selv.size > 0:
            rnn_states_critic[done_mask] = 0.0

        # --- masks: done인 자리만 0으로 ---
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[done_mask] = 0.0                 # (k, 1) 타깃에 브로드캐스트로 안전하게 들어감

        # --- share_obs ---
        if self.use_centralized_V:
            # obs: (n_threads, n_agents, obs_dim) -> (n_threads, n_agents*obs_dim)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # 그리고 다시 (n_threads, n_agents, n_agents*obs_dim)로 복제
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs, obs,
            rnn_states, rnn_states_critic,
            actions, action_log_probs, values,
            rewards, masks
        )
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(self.save_dir, episode)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)