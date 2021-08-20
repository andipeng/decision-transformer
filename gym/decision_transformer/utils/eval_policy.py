import unittest
import mj_envs
from mjrl.utils.gym_env import GymEnv
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg, eval_test_goals
from decision_transformer.evaluation.rollout import sample_paths
from decision_transformer.models.decision_transformer import DecisionTransformer

import click
import pickle
import numpy as np
import torch

DESC = '''
Helper script to eval success policy.\n
USAGE:\n
    $ python decision_transformer/utils/eval_policy.py --policy policy.pickle \n
'''

eval_horizons = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--policy', type=str, help='path to the trained GCBC policy', required=True)
@click.option('--train_data', type=str, help='absolute path of the training data (for input normalization)', default='data/kitchen-nohindsight/subfolder.pkl')
@click.option('--test_data', type=str, help='absolute path of the test data', default=None)
@click.option('--horizon', type=int, help='horizon to evaluate rollout on (0 = full sweep)', default=0)
@click.option('--num_traj', type=int, help='number of rollouts to eval per env', default=10)
@click.option('--env', type=str, help='environment to load', default='kitchen-v3')
@click.option('--target_rew', type=int, help='target reward to condition on', default=0)
@click.option('--scale', type=float, help='scaling of reward', default=1.)
@click.option('--mode', type=str, help='normal or delayed for sparse reward settings', default='normal')
@click.option('--seed', type=int, help='seed for env', default=123)

def main(policy, train_data, test_data, horizon, num_traj, env, target_rew, scale, mode, seed):
    pi = torch.load(policy)

    if horizon == 0:
        horizon = eval_horizons
    
    # if not provided, test data is same as training data
    if test_data is None:
        test_data = train_data
    test_data = pickle.load(open(test_data, 'rb'))
    train_data = pickle.load(open(train_data, 'rb'))

    eval_rewards, eval_success_rates, eval_reward, eval_success = eval_test_goals(
                                                                    model=pi,
                                                                    train_data=train_data,
                                                                    test_data=test_data,
                                                                    eval_shift=horizon,
                                                                    eval_env=env,
                                                                    num_traj=num_traj,
                                                                    target_rew=target_rew,
                                                                    scale=scale,
                                                                    mode=mode,
                                                                    seed=seed)
    
    for i in range(len(eval_rewards)):
        print("Policy reward: %f, success rate: %f" % (eval_rewards[i], eval_success_rates[i]))

if __name__ == '__main__':
    main()