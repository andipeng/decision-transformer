import unittest
import mj_envs
from mjrl.utils.gym_env import GymEnv
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg, sample_paths
from decision_transformer.models.decision_transformer import DecisionTransformer
from gcdt.utils.dataloader import simple_hindsight_relabel
from gcdt.utils.evaluate import eval_test_envs, eval_test_goals

import click
import pickle
import numpy as np
import torch

DESC = '''
Helper script to eval success policy.\n
USAGE:\n
    $ python decision_transformer/utils/eval_policy.py --policy policy.pickle \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env', type=str, help='environment to load', default='kitchen-v3')
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--data', type=str, help='absolute path of the training data (for input normalization)', default='data/kitchen-expert-v2.pkl')
@click.option('--target_rew', type=int, help='target reward to condition on', default=0)
@click.option('--scale', type=float, help='scaling of reward', default=1.)
@click.option('--num_episodes', type=int, help='number of episodes to eval on', default=3)
@click.option('--max_ep_len', type=int, help='max episode length', default=300)
@click.option('--mode', type=str, help='normal or delayed for sparse reward settings', default='normal')
@click.option('--seed', type=int, help='seed for env', default=123)

def main(env, policy, data, target_rew, scale, max_ep_len, num_episodes, mode, seed):
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
    with torch.no_grad():
        paths = sample_paths(
                num_traj=num_episodes,
                env=e,
                state_dim=e.observation_space.shape[0],
                act_dim=e.action_space.shape[0],
                model=model,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=target_rew/scale,
                mode='normal',
                state_mean=state_mean,
                state_std=state_std,
                base_seed=seed,
                #device='cpu',
            )

    print(paths)

if __name__ == '__main__':
    main()