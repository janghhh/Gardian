import wandb
import os
import numpy as np
import torch
import time
from collections import defaultdict
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

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
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """메인 루프. 총괄."""
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # 에피소드별 상세 보상 기록을 위한 딕셔너리 초기화
            ep_rewards_detailed = defaultdict(list)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions)
                
                # infos에서 상세 보상 내역을 추출하여 기록
                for info in infos:
                    if "rewards" in info:
                        for k, v in info["rewards"].items():
                            ep_rewards_detailed[k].append(v)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            self.compute()
            train_infos = self.train()
            
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            if (episode % self.log_interval == 0):
                end = time.time()
                print(f"\n Env-{self.env_name} ... FPS {int(total_num_steps / (end - start))}.")
                
                # --- 로깅 부분 수정 ---
                # 전체 평균 보상 로깅
                total_ep_reward = np.mean(self.buffer.rewards[self.buffer.rewards != 0])
                train_infos['episode_reward'] = total_ep_reward
                
                # 상세 보상 내역 로깅
                env_infos = {}
                for k, v in ep_rewards_detailed.items():
                    env_infos[f"reward/{k}"] = v # wandb에서 그룹으로 묶어보기 위해 이름 변경
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # 평가
            if (self.use_eval and episode % self.eval_interval == 0):
                self.eval(total_num_steps)

            # 모델 저장
            if (episode % self.save_interval == 0 or episode == episodes - 1):
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
        """정책을 이용해 행동을 결정하고, 다음 RNN 상태 등을 반환합니다."""
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
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

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """환경으로부터 받은 데이터를 버퍼에 삽입."""
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        rewards = np.expand_dims(rewards, -1)
        # RNN을 위한 마스크 처리
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # share_obs 처리
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)
    
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
