import torch
import numpy as np
from mjrl.utils.gym_env import GymEnv
from mjrl.utils import tensor_utils

def sample_paths(
        num_traj,
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
        base_seed=None,
        env_kwargs=None,
        *args,
        **kwargs,
    ):

    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError
    
    if base_seed is not None:
        env.set_seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    max_ep_len = min(max_ep_len, env.horizon)

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device) + 1e-6

    paths = []
    for ep in range(num_traj):
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            env.set_seed(seed)
            np.random.seed(seed)
        
        path_obs=[]
        path_acts=[]
        path_rews=[]
        #path_agent_infos = []
        path_env_infos = []

        state = env.reset()
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = target_return
        target_return_tensor = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        sim_states = []

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):
            env_info_base = env.get_env_infos()

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return_tensor.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()
            path_acts.append(action)

            state, reward, done, env_info_step = env.step(action)
            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            path_obs.append(state)
            path_rews.append(reward)
            path_env_infos.append(env_info)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if mode != 'delayed':
                pred_return = target_return_tensor[0,-1]# - (reward/scale)
            else:
                pred_return = target_return_tensor[0,-1]
            target_return_tensor = torch.cat(
                [target_return_tensor, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            if done:
                break
        
        path = dict(
            observations=np.array(path_obs),
            actions=np.array(path_acts),
            rewards=np.array(path_rews),
            #agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(path_env_infos),
            terminated=done
        )
        paths.append(path)
    
    del(env)
    return paths