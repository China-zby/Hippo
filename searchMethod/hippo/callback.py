import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class TestActorCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=100, verbose=1):
        super(TestActorCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_step = 0
        self.video_list = [0] # , 200, 400, 600, 800, 999
        self.save_path = "./ppo_tensorboard/eval_result"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.bast_metric = 0

    def _on_step(self) -> bool:
        self.eval_step += 1
        if self.eval_step % self.eval_freq == 0:
            self.test_actor()
        return True

    def test_actor(self):
        save_result_path = os.path.join(self.save_path, f"step_{self.eval_step}")
        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)
        # print(f"Testing actor at step {self.eval_step}")
        total_acc, total_lat = 0, 0
        for video_id in self.video_list:    
            obs, _ = self.eval_env.reset_with_videoid(video_id)
            done = False
            truncated = False
            total_reward = 0
            while not done and not truncated:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, done, truncated, _info = self.eval_env.step(action)
                total_reward += rewards
            # print(f"Total reward: {total_reward}")
            plt.scatter([res[0] for res in self.eval_env.pareto_set],
                        [res[1] for res in self.eval_env.pareto_set], c='r',
                        marker='x', label=f'pareto set {video_id}')
            plt.savefig(os.path.join(save_result_path, f"pareto_set_{video_id}.png"))
            plt.close()
            plt.cla()
            plt.clf()
            with open(os.path.join(save_result_path, f"pareto_set_{video_id}.txt"), "w") as f:
                mean_acc, mean_lat = [], []
                for res in self.eval_env.pareto_set:
                    f.write(f"{res[0]} {res[1]} {res[2]}\n")
                    mean_acc.append(res[0])
                    mean_lat.append(res[1])
                mean_acc = sum(mean_acc) / len(mean_acc)
                mean_lat = sum(mean_lat) / len(mean_lat)
                f.write(f"{mean_acc} {mean_lat} {00000000}\n")
                self.logger.record(f'eval/mean_acc_{video_id}', mean_acc)
                self.logger.record(f'eval/mean_lat_{video_id}', mean_lat)
                self.logger.record(f'eval/pset_num_{video_id}', len(self.eval_env.pareto_set))
                total_acc += mean_acc
                total_lat += mean_lat
        total_acc = total_acc / len(self.video_list)
        total_lat = total_lat / len(self.video_list)
        
        total_metric = total_acc * 0.5 + total_lat * 0.5
        if total_metric > self.bast_metric:
            self.bast_metric = total_metric
            self.model.save(os.path.join(self.save_path, "best_model"))    

class CustomCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            total_rewards = 0
            done = False
            obs = self.eval_env.reset()
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, done, info = self.eval_env.step(action)
                total_rewards += rewards

            print(
                f"Total rewards at step {self.num_timesteps}: {total_rewards}")

        return True
    
class SaveRewardsCallback(BaseCallback):
    def __init__(self, check_freq, save_path='training_rewards.png'):
        super(SaveRewardsCallback, self).__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.rewards = []

    def _on_step(self) -> bool:
        # 每check_freq步收集奖励
        if self.n_calls % self.check_freq == 0:
            rewards = self.training_env.get_attr('rewards')  # 获取环境奖励列表
            mean_reward = np.mean([np.mean(reward) for reward in rewards])
            self.rewards.append(mean_reward)
        return True

    def _on_training_end(self):
        # 绘制并保存奖励图像
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.title('Training Rewards Over Time')
        plt.savefig(self.save_path)
        plt.close()  # 关闭图形，防止其显示在notebook或python脚本执行环境中
        plt.cla()
        plt.clf()