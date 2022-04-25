#  Copyright: (C) ETAS GmbH 2019. All rights reserved.

import sys
import os
import csv
import random
import numpy as np
import gym
import pybullet_envs
import torch
from collections import deque

from common import helper
from rllib.model_sac import FlowSAC
from common.helper import update_linear_schedule
from common.eval_policy import eval_env
from rllib.behavioral_cloning import BehavioralCloning


def pretrain_bc(num_bc_updates=1000, lr_bc=1e-4, bc_batch_size=256, use_mlp=False,
                grad_clip_val=10., weight_decay=1e-5):
    use_linear_lr_decay_bc = False
    weight_clipping = False
    weight_clip_val = 1.
    losses = deque(maxlen=100)
    agent = BehavioralCloning(flow_policy, lr_bc, weight_decay,
                              flow_model=(not use_mlp), grad_clip_val=grad_clip_val)
    for j in range(num_bc_updates):
        if use_linear_lr_decay_bc:
            update_linear_schedule(agent.optimizer, j, num_bc_updates, lr_bc)

        indices = np.array(random.sample(range(0, train_state.shape[0] - 1), bc_batch_size))
        indices = np.sort(indices)
        actions_trajs = torch.tensor(train_action[indices, :], dtype=torch.float).to(device)
        state_trajs = torch.tensor(train_state[indices, :], dtype=torch.float).to(device)

        loss = agent.update(state_trajs, actions_trajs)

        if weight_clipping:
            for p in flow_policy.get_policy_parameter():
                p.data.clamp_(-weight_clip_val, weight_clip_val)

        losses.append(loss)

        if j % 25 == 0:
            print("Update Steps: ", j)
            print("Mean BC Loss: ", np.mean(losses))

            indices = np.array(random.sample(range(0, valid_state.shape[0] - 1), bc_batch_size))
            indices = np.sort(indices)
            test_actions = torch.tensor(valid_action[indices, :], dtype=torch.float).to(device)
            test_state = torch.tensor(valid_state[indices, :], dtype=torch.float).to(device)
            test_loss = agent.estimate_error(test_state, test_actions)
            print("Mean Test Loss: ", np.mean(test_loss))
        if j % 100 == 0:
            flow_policy.save(os.path.join(output_dir, env_name + "_d.pt"))


if __name__ == '__main__':
    # Training Parameter
    num_bc_updates = 10000
    bc_batch_size = 256
    lr_bc = 1e-4
    grad_clip_val = 10.
    weight_decay = 1e-5
    visualize = False
    use_tanh = True

    seed = 3
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir("./Experiments"):
        os.mkdir("./Experiments")
    output_dir = os.path.join("./Experiments", "test")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    env_name = "AntBulletEnv-v0"
    load_model = False

    # Simulation and expert data set
    expert_file_dir = "data"
    expert_filename = env_name + "expert"
    n_trajectories = 10
    train_state, train_action, valid_state, valid_action, _, _ = helper.load_hdf5_data_wns(expert_file_dir,
                                                                                           expert_filename,
                                                                                           n_trajectories)

    # Build Gym
    train_action_scale = np.max(np.abs(train_action))
    test_action_scale = np.max(np.abs(valid_action))
    action_scale_ = max(train_action_scale, test_action_scale)
    use_norm = False
    env = gym.make(env_name)

    # Get state and action dimensions, action scale
    num_observations = env.observation_space.shape[0]
    if env.action_space.__class__.__name__ == "Discrete":
        raise NotImplementedError()
    else:
        num_action = env.action_space.shape[0]
        action_scale = env.action_space.high[0]
        print("action scale from data: ", action_scale_)
        print("action scale from environment: ", env.action_space.high[0])
        print("action scale used: ", action_scale)
        if num_action == 1:
            raise NotImplementedError()

    # build policy
    n_flows_policy = 16    # This is in general a good depth
    policy_flow_hidden = 64  # increase this if neccessary
    exp_clamping = 4.      # This helps to have a stable policy training, higher values enable the policy to have more complex distributions
    hidden_size_cond = 64  # increase this if neccessary
    con_dim_features = 8   # increase this if neccessary
    flow_policy = FlowSAC(num_observations=num_observations, num_action=num_action,
                          n_flows=n_flows_policy, flow_hidden=policy_flow_hidden,
                          exp_clamping=exp_clamping, hidden_size_cond=hidden_size_cond,
                          act_limit=action_scale, con_dim_features=con_dim_features,
                          use_tanh=use_tanh, device=device).to(device)

    if load_model:
        model_location = os.path.join(output_dir, env_name + "_d.pt")
        if os.path.isfile(model_location):
            flow_policy.load(model_location)
        else:
            print("No policy model available at " + model_location)

    # train policy using BC
    pretrain_bc(num_bc_updates=num_bc_updates, lr_bc=lr_bc, bc_batch_size=bc_batch_size, use_mlp=False,
                grad_clip_val=grad_clip_val, weight_decay=weight_decay)

    # evaluate policy in environment
    print("Start policy evaluation... ")
    flow_policy.test_mode = True
    if visualize:
        eval_env(env, flow_policy, visualize=visualize, use_additional_normal=False)
    test_reward = [eval_env(env, flow_policy, use_additional_normal=False,
                            output_dir=output_dir) for _ in range(10)]
    print("Test Reward: ", np.mean(test_reward))
    csv_path = os.path.join(output_dir, env_name + "_" + str(int(np.mean(test_reward))) + "_metrics.csv")
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["mean reward", "{:.6f}".format(np.mean(test_reward))])
        writer.writerow(test_reward)

    env.close()

    print("done!")
    quit()
