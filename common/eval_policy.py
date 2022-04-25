#  Copyright: (C) ETAS GmbH 2019. All rights reserved.

import os
import numpy as np
import csv
import random
import gym
import torch


def eval_env(env, actor_critic, action_scale=None, visualize=False, max_env_steps=2000, use_additional_normal=False,
             output_dir="./", epoch=1):
    '''Evaluate arbitrary OPENAI Gym Environment by collecting its reward'''
    device = actor_critic.device
    is_discrete = False
    if env.action_space.__class__.__name__ == "Discrete":
        is_discrete = True
    state = env.reset()
    if visualize:
        env.render()
    done = False
    total_reward = 0
    env_steps = 0
    actions = []
    states = []
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = actor_critic(state)
        action = dist.sample()
        if is_discrete:
            # _, action = action.max(dim=1)
            action = action.squeeze(-1)
        else:
            if use_additional_normal:
                action = action[:, 0].unsqueeze(1)
            if action_scale is not None:
                action = torch.clamp(action, -action_scale, action_scale)
        # np_state = state.cpu().squeeze(0).numpy()
        # if np_state[0] < 0. and np_state[1] < 0.:
        #     action = 0
        # elif np_state[1] >= 0.:  # np_state[0] >= 0. and
        #     action = 2
        # else:
        #     action = 0
        action = action.cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        states.append(next_state)
        actions.append(action)
        if visualize: env.render()
        state = next_state
        total_reward += reward
        env_steps +=1
        if env_steps > max_env_steps:
            done = True

    return total_reward  #, actions
