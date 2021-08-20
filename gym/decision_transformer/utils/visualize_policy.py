import gym
import click 
import os
import gym
import numpy as np
import pickle
import torch
import skvideo.io

import mj_envs
from mjrl.utils.gym_env import GymEnv
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer

DESC = '''
Helper script to eval/visualize policy.\n
USAGE:\n
    Visualizes policy on the specified env. \n
    $ python decision_transformer/utils/visualize_policy.py --policy policy.pickle \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env', type=str, help='environment to load', default='kitchen-v3')
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--data', type=str, help='absolute path of the training data (for input normalization)', default='data/kitchen-nohindsight/subfolder.pkl')
@click.option('--target_rew', type=int, help='target reward to condition on', default=0)
@click.option('--scale', type=float, help='scaling of reward', default=1.)
@click.option('--num_episodes', type=int, help='number of episodes to eval on', default=1)
@click.option('--max_ep_len', type=int, help='max episode length', default=300)
@click.option('--render', type=bool, help='render policy', default=True)
@click.option('--mode', type=str, help='normal or delayed for sparse reward settings', default='normal')

def main(env, policy, data, target_rew, scale, max_ep_len, num_episodes, render, mode):
    e = GymEnv(env)
    model = torch.load(policy)
    trajectories = pickle.load(open(data, 'rb'))

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    returns, lengths = [],[]
    for _ in range(num_episodes):
        with torch.no_grad():
            ret, length = evaluate_episode_rtg(
                e,
                e.observation_space.shape[0],
                e.action_space.shape[0],
                model,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=target_rew/scale,
                mode='noise',
                state_mean=state_mean,
                state_std=state_std,
                device='cpu',
                render=render,
            )
            returns.append(ret)
            lengths.append(length)

    print('return_mean: ', np.mean(returns))
    print('return_std ', np.std(returns))
    print('length_mean ',np.mean(lengths))
    print('length_std ',np.std(lengths))

if __name__ == '__main__':
    main()