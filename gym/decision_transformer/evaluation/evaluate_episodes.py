import numpy as np
import torch
import pickle
import random
from mjrl.utils.gym_env import GymEnv
from mjrl.utils import tensor_utils
from gcdt.utils.dataloader import simple_hindsight_relabel
from decision_transformer.evaluation.rollout import sample_paths

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        render=False,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.01, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)
        if render:
            env.render()

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1]# - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

def eval_test_goals(model, train_data, test_data, eval_shift, eval_env, target_rew, scale, num_traj, mode, seed):
    eval_rewards, eval_success_rates = ([] for _ in range(2))
    e = GymEnv(eval_env)

    model.eval()

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in train_data:
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

    # we evaluate on all of the eval shift windows specified
    for i in eval_shift:
        # relabel test data to get the appropriate shifted goal
        shifted_test_data = simple_hindsight_relabel(e, test_data, i)

        # we eval on (randomly selected) shifted goals (with eval horizon) from test_data
        eval_paths = []
        for _ in range(num_traj):
            e.reset()
            eval_demo = random.choice(shifted_test_data)    # select a random demo to eval on
            initial_obs = eval_demo['observations'][0]      # first timestep of demo
            eval_goal = initial_obs[e.env.env.key_idx['goal']]  # the shifted (relabeled) goal
            #e.env.env.set_state(eval_demo['init_qpos'], eval_demo['init_qvel'])
            e.env.env.set_goal(eval_goal)

            with torch.no_grad():
                eval_path = sample_paths(
                        num_traj=1,
                        env=e,
                        state_dim=e.observation_space.shape[0],
                        act_dim=e.action_space.shape[0],
                        model=model,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode='normal',
                        state_mean=state_mean,
                        state_std=state_std,
                        base_seed=seed,
                        device='cpu',
                    )
            eval_paths.extend(eval_path)
        eval_reward = np.mean([np.sum(p['rewards']) for p in eval_paths])
        try:
            eval_success = e.env.env.evaluate_success(eval_paths)
        except:
            eval_success = 0.0
        eval_rewards.append(eval_reward)
        eval_success_rates.append(eval_success)
            
    # compute mean statistics
    eval_reward = np.mean(eval_rewards)
    eval_success = np.mean(eval_success_rates)

    return eval_rewards, eval_success_rates, eval_reward, eval_success