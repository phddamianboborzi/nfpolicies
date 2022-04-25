#  Copyright: (C) ETAS GmbH 2019. All rights reserved.
import os
import numpy as np
import torch
import torch.nn as nn
import gym
import h5py
from scipy.special import rel_entr
from scipy.spatial import distance
import csv
from collections import deque

features = {
    0: "lat_pos",
    1: "l_ind",
    2: "v",
    3: "a",
    4: "heading",
    5: "length",
    6: "width",
    7: "lat_change",
    8: "head_change",
    9: "dis_2_right",
    10: "dis_2_left",
    11: "last_acc",
    12: "last_lc",
    13: "lon_dis_lead",
    14: "lat_dis_lead",
    15: "v_lead",
    16: "a_lead",
    17: "length_lead",
    18: "width_lead",
    19: "ttc",
    20: "lon_dis_1",
    21: "lat_dis_1",
    22: "l_ind_1",
    23: "v_1",
    24: "length_1",
    25: "width_1",
    26: "lon_dis_2",
    27: "lat_dis_2",
    28: "l_ind_2",
    29: "v_2",
    30: "length_2",
    31: "width_2",
    32: "lon_dis_3",
    33: "lat_dis_3",
    34: "l_ind_3",
    35: "v_3",
    36: "length_3",
    37: "width_3",
    38: "lon_dis_4",
    39: "lat_dis_4",
    40: "l_ind_4",
    41: "v_4",
    42: "length_4",
    43: "width_4",
    44: "lon_dis_5",
    45: "lat_dis_5",
    46: "l_ind_5",
    47: "v_5",
    48: "length_5",
    49: "width_5"}


def make_env(seed, rank, env_name):
    def _thunk():
        env = gym.make(env_name)
        seed_val = 2 * (seed + rank) + 1
        env.seed(seed_val)
        env.reset(env_id=rank)
        return env

    return _thunk


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


def load_pandas_data(load_dir, seq_length=1, filename="feature_pandas", skip_step=1):  # feature_pandas_interest
    import pickle
    n_feature = 49 

    print('loading data...')
    d = []
    with open(os.path.join(load_dir, filename), 'rb') as fp:
        data = pickle.load(fp)

    total_idx = 0
    for traj in data:
        idx = 0
        length = np.shape(traj)[0]
        while (idx + seq_length) < length:
            d.append(traj[idx:idx + seq_length, :])
            idx += skip_step
            total_idx += skip_step
        if total_idx > 10000:
            break
    d = np.asarray(d, dtype="float32")
    np.random.shuffle(d)
    s = d[:, :, 0:n_feature]
    a = d[:, :, -2:]

    train_size = int(np.shape(s)[0] * 0.8)
    train_x, valid_x = s[0:train_size, :, :], s[train_size:, :, :]
    train_y, valid_y = a[0:train_size, :, :], a[train_size:, :, :]
    # train_x = np.reshape(train_x, (train_x.shape[0], -1))
    # valid_x = np.reshape(valid_x, (valid_x.shape[0], -1))
    # train_y = np.reshape(train_y, (train_y.shape[0], -1))
    # valid_y = np.reshape(valid_y, (valid_y.shape[0], -1))

    print('loading data completed, num of trajectory:', train_size)
    # train_file = os.path.join(load_dir, filename + ".hdf5")
    # if not os.path.isfile(train_file):
    #     print("generate hdf5 file")
    #     hdf5_store = h5py.File(train_file, "a")
    #     train_x_h = hdf5_store.create_dataset("train_x", train_x.shape)  # , compression="gzip"
    #     train_x_h[:] = train_x[:]
    #     train_y_h = hdf5_store.create_dataset("train_y", train_y.shape)
    #     train_y_h[:] = train_y[:]
    #     valid_x_h = hdf5_store.create_dataset("valid_x", valid_x.shape)  # , compression="gzip"
    #     valid_x_h[:] = valid_x[:]
    #     valid_y_h = hdf5_store.create_dataset("valid_y", valid_y.shape)
    #     valid_y_h[:] = valid_y[:]
    return train_x, train_y, valid_x, valid_y


def load_hdf5_data(load_dir, filename="feature_pandas"):
    hdf5_store = h5py.File(os.path.join(load_dir, filename + ".hdf5"), "r")
    train_x = hdf5_store["train_x"]
    train_y = hdf5_store["train_y"]
    valid_x = hdf5_store["valid_x"]
    valid_y = hdf5_store["valid_y"]

    train_state = train_x[:].reshape((-1, train_x.shape[-1]))
    train_action = train_y[:].reshape((-1, train_y.shape[-1]))
    valid_state = valid_x[:].reshape((-1, valid_x.shape[-1]))
    valid_action = valid_y[:].reshape((-1, valid_y.shape[-1]))
    return train_state, train_action, valid_state, valid_action


def load_hdf5_data_wt(load_dir, filename="feature_pandas"):
    hdf5_store = h5py.File(os.path.join(load_dir, filename + ".hdf5"), "r")
    train_x = hdf5_store["train_x"]
    train_y = hdf5_store["train_y"]
    valid_x = hdf5_store["valid_x"]
    valid_y = hdf5_store["valid_y"]
    train_t = hdf5_store["train_t"]
    valid_t = hdf5_store["valid_t"]
    return train_x, train_y, valid_x, valid_y, train_t, valid_t


def load_hdf5_data_wns(load_dir, filename="feature_pandas", n_trajectories=-1):
    hdf5_store = h5py.File(os.path.join(load_dir, filename + ".hdf5"), "r")
    train_x = hdf5_store["train_x"]
    train_y = hdf5_store["train_y"]
    valid_x = hdf5_store["valid_x"]
    valid_y = hdf5_store["valid_y"]
    train_n_x = hdf5_store["train_n_x"]
    valid_n_x = hdf5_store["valid_n_x"]

    if n_trajectories > 0:
        train_x = train_x[:n_trajectories, :, :]
        train_y = train_y[:n_trajectories, :, :]
        valid_x = valid_x[:n_trajectories, :, :]
        valid_y = valid_y[:n_trajectories, :, :]
        train_n_x = train_n_x[:n_trajectories, :, :]
        valid_n_x = valid_n_x[:n_trajectories, :, :]

    train_state = train_x[:].reshape((-1, train_x.shape[-1]))
    train_action = train_y[:].reshape((-1, train_y.shape[-1]))
    valid_state = valid_x[:].reshape((-1, valid_x.shape[-1]))
    valid_action = valid_y[:].reshape((-1, valid_y.shape[-1]))
    train_next_state = train_n_x[:].reshape((-1, train_n_x.shape[-1]))
    valid_next_state = valid_n_x[:].reshape((-1, valid_n_x.shape[-1]))

    return train_state, train_action, valid_state, valid_action, train_next_state, valid_next_state


def eval_env(env, model, test_steps=500, device="cpu", state_scaler=None, use_gru=False, deterministic=True):
    state = env.reset()
    state = np.stack(state)
    model.reset_hidden(batch_size=state.shape[0])
    if state_scaler is not None:
        state = state_scaler.transform(state)
    total_reward = []
    speeds = []
    accs = []
    n_ls = 0
    n_hard_brake = 0
    turn_rates = []
    jerks = []
    actions = []
    # states = []
    # states.append(state)
    collisions = 0
    total_n_traj = 0
    n_traj = 0
    total_driven_dist = 0
    driven_dist = 0
    ittc = []
    l_s = []
    d_a = []
    sba = 0
    sflc = 0
    for i in range(test_steps):
        state = torch.FloatTensor(state).to(device)
        dist, _ = model(state)
        # if deterministic:
        #     action = dist.mode().detach().clone().cpu().numpy()
        # else:
        action = dist.sample().clone().cpu().numpy()
        actions.append(action.copy())
        state, reward, done, infos = env.step(action)
        if state_scaler is not None:
            state = np.stack(state)
            state = state_scaler.transform(state)
        n_traj = 0
        driven_dist = 0
        for it in range(env.nenvs):
            info = infos[it]
            collision_occoured, tmp_speed, tmp_acc, tmp_n_ls, tmp_n_hard_brake, \
            tmp_turn_rates, tmp_jerks, n_traj_se, tmp_collisions, driven_dist_se, tmp_ittc, \
            tmp_l_s, tmp_d_a, tmp_sflc, tmp_sba = gather_information(info)
            speeds.extend(tmp_speed)
            accs.extend(tmp_acc)
            n_ls += tmp_n_ls
            # collisions += tmp_collisions
            n_hard_brake += tmp_n_hard_brake
            sflc += tmp_sflc
            sba += tmp_sba
            turn_rates.extend(tmp_turn_rates)
            jerks.extend(tmp_jerks)
            ittc.extend(tmp_ittc)
            l_s.extend(tmp_l_s)
            d_a.extend(tmp_d_a)
            total_reward.extend(reward)
            n_traj += n_traj_se
            driven_dist += driven_dist_se
            if collision_occoured:
                collisions += 1
                total_n_traj += n_traj
                total_driven_dist += driven_dist_se
        # if collision_occoured:
        #     collisions += 1
        #     state = env.reset()
        #     state = np.stack(state)
        #     total_n_traj += n_traj
        #     total_driven_dist += driven_dist
        # if False not in done:
        #     state = env.reset()
        #     state = np.stack(state)
        #     total_n_traj += n_traj
        #     total_driven_dist += driven_dist
    total_n_traj += n_traj
    total_driven_dist += driven_dist
    return np.sum(total_reward), speeds, accs, collisions, np.concatenate(actions, 0), \
           total_n_traj, n_ls, n_hard_brake, turn_rates, jerks, total_driven_dist, ittc, l_s, d_a, sflc, sba


def gather_information(info):
    speeds = info["speed"]
    accs = info["acceleration"]
    collision_occoured = info["ego_collision"]
    collisions = info["collision_ids"]
    lane_change_ids = info["lane_changes"]
    hard_brake_ids = info["hard_brake"]
    turn_rates = info["turn_rate"]
    jerks = info["jerk"]
    number_trajectories = info["n_traj"]
    driven_dist = info["total_dist"]
    ittc = info["ittc"]
    l_s = info["lane_and_speed"]
    d_a = info["dist_acc"]
    sflc_active = info["sflc_active"]
    sba_active = info["sba_active"]
    return collision_occoured, speeds, accs, len(lane_change_ids), \
           len(hard_brake_ids), turn_rates, jerks, number_trajectories, len(collisions), driven_dist, ittc, l_s, d_a, sflc_active, sba_active


class SumoJSD():
    def __init__(self, writer, clear_n_epoch=5, dir_feature='./data/', run_dir="./",
                 metric_file="highD_training_metrics.csv"):
        self.writer = writer
        self.run_dir = run_dir
        self.csv_path = os.path.join(run_dir, metric_file)
        with open(self.csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["episode", "total_km", "n_lc", "hard_brake", "collisions",
                             "jsd_speed", "jsd_all_accs", "jsd_all_turn_rates",
                             "jsd_all_jerks", "jsd_all_ittc", "jsd_all_lane_and_speed", "jsd_all_dist_acc"])

        self.data_all_speeds = np.load(os.path.join(dir_feature, "all_speeds" + ".npy"))
        self.data_all_accs = np.load(os.path.join(dir_feature, "all_accs" + ".npy"))
        self.data_all_ittc = np.load(os.path.join(dir_feature, "all_ittc" + ".npy"))
        self.data_all_dist_acc = np.load(os.path.join(dir_feature, "all_dist_acc" + ".npy"))
        self.data_all_lane_and_speed = np.load(os.path.join(dir_feature, "all_lane_and_speed" + ".npy"))
        self.data_all_turn_rates = np.load(os.path.join(dir_feature, "all_turn_rates" + ".npy"))
        self.data_all_jerks = np.load(os.path.join(dir_feature, "all_jerks" + ".npy"))
        self.speeds = []  # deque(maxlen=10000)
        self.accs = []  # deque(maxlen=10000)
        self.ittc = []  # deque(maxlen=10000)
        self.l_s = []  # deque(maxlen=10000)
        self.d_a = []  # deque(maxlen=10000)
        self.turn_rates = []
        self.jerks = []
        self.collisions = 0
        self.n_lane_change = 0
        self.n_hard_brake = 0
        self.total_driven_dist = 0
        self.clear_n_epoch = clear_n_epoch

    def append_infos(self, infos):
        for info in infos:
            self.speeds.extend(info["speed"])
            self.accs.extend(info["acceleration"])
            self.turn_rates.extend(info["turn_rate"])
            self.ittc.extend(info["ittc"])
            self.l_s.extend(info["lane_and_speed"])
            self.d_a.extend(info["dist_acc"])
            self.n_lane_change += len(info["lane_changes"])
            self.n_hard_brake += len(info["hard_brake"])
            self.jerks.extend(info["jerk"])
            if info["ego_collision"]:
                self.collisions += 1

    def calculate_distance(self, infos):
        for info in infos:
            self.add_drive_dist(info["total_dist"])

    def add_drive_dist(self, distance):
        self.total_driven_dist += distance

    def clear_all(self):
        self.speeds = []
        self.accs = []
        self.turn_rates = []
        self.ittc = []
        self.l_s = []
        self.d_a = []
        self.jerks = []
        self.collisions = 0

    def estimate_jsd(self, epoch):
        if self.total_driven_dist > 0:
            total_km = float(self.total_driven_dist) / 1000.0
            print("collision_rate: ", self.collisions / total_km)
            print("lane_change_rate: ", self.n_lane_change / total_km)
            self.writer.add_scalar('rl_training/lane_change_rate', self.n_lane_change / total_km, epoch)
            self.writer.add_scalar('rl_training/hard_brake_rate', self.n_hard_brake / total_km, epoch)
            self.writer.add_scalar('rl_training/collision_rate', self.collisions / total_km, epoch)
        else:
            total_km = 0

        if len(self.speeds) > 0:
            jsd_speed = calculate_js_dis(self.data_all_speeds, self.speeds)
            jsd_all_accs = calculate_js_dis(self.data_all_accs, self.accs)
            jsd_all_turn_rates = calculate_js_dis(self.data_all_turn_rates, self.turn_rates)
            jsd_all_ittc = calculate_js_dis(self.data_all_ittc, self.ittc)
            jsd_all_lane_and_speed = calculate_js_dis(self.data_all_lane_and_speed, self.l_s)
            jsd_all_dist_acc = calculate_js_dis(self.data_all_dist_acc, self.d_a)
            jsd_all_jerks = calculate_js_dis(self.data_all_jerks, self.jerks)
            print("jsd_speed: ", jsd_speed)
            print("jsd_all_accs: ", jsd_all_accs)
            print("jsd_all_turn_rates: ", jsd_all_turn_rates)
            print("jsd_all_ittc: ", jsd_all_ittc)
            print("jsd_all_dist_acc: ", jsd_all_dist_acc)
            print("jsd_all_lane_and_speed: ", jsd_all_lane_and_speed)
            print("jsd_all_jerks: ", jsd_all_jerks)
            self.writer.add_scalar('rl_training/jsd_speed', jsd_speed, epoch)
            self.writer.add_scalar('rl_training/jsd_all_accs', jsd_all_accs, epoch)
            self.writer.add_scalar('rl_training/jsd_all_turn_rates', jsd_all_turn_rates, epoch)
            self.writer.add_scalar('rl_training/jsd_all_ittc', jsd_all_ittc, epoch)
            self.writer.add_scalar('rl_training/jsd_all_dist_acc', jsd_all_dist_acc, epoch)
            self.writer.add_scalar('rl_training/jsd_all_lane_and_speed', jsd_all_lane_and_speed, epoch)
            self.writer.add_scalar('rl_training/jsd_all_jerks', jsd_all_jerks, epoch)

            self.write_csv(epoch, total_km, self.n_lane_change, self.n_hard_brake, self.collisions,
                      jsd_speed, jsd_all_accs, jsd_all_turn_rates, jsd_all_jerks, jsd_all_ittc,
                      jsd_all_lane_and_speed, jsd_all_dist_acc)

        if self.clear_n_epoch > 0 and epoch > 1 and epoch % self.clear_n_epoch == 0:
            self.clear_all()

    def write_csv(self, episode, total_km, total_n_ls, total_N_hard_brake, all_collisions,
                  jsd_speed, jsd_all_accs, jsd_all_turn_rates, jsd_all_jerks, jsd_all_ittc,
                  jsd_all_lane_and_speed, jsd_all_dist_acc):
        with open(self.csv_path, "a") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(["{:.6f}".format(episode),
                             "{:.6f}".format(total_km),
                             "{:.6f}".format(total_n_ls),
                             "{:.6f}".format(total_N_hard_brake),
                             "{:.6f}".format(all_collisions),
                             "{:.6f}".format(jsd_speed),
                             "{:.6f}".format(jsd_all_accs),
                             "{:.6f}".format(jsd_all_turn_rates),
                             "{:.6f}".format(jsd_all_jerks),
                             "{:.6f}".format(jsd_all_ittc),
                             "{:.6f}".format(jsd_all_lane_and_speed),
                             "{:.6f}".format(jsd_all_dist_acc)])


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, lr_min=1e-5):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    if lr < lr_min:
        lr = lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Necessary for KFAC implementation.
# See: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class DiagStd(nn.Module):
    def __init__(self, num_outputs):
        super(DiagStd, self).__init__()
        self._bias = nn.Parameter(torch.eye(num_outputs))

    def forward(self, x):
        return x + self._bias


def calculate_kl_div(target, approx):
    min = np.min(target)
    max = np.max(target)
    eps = 1e-8

    target_hist, target_bin_edges = np.histogram(target, bins=np.arange(min, max, (max-min)/100), density=True)
    p = (target_hist+eps) * np.diff(target_bin_edges)

    approx_hist, approx_bin_edges = np.histogram(approx, bins=np.arange(min, max, (max - min) / 100), density=True)
    q = (approx_hist+eps) * np.diff(approx_bin_edges)

    kl_pq = rel_entr(p, q)
    return sum(kl_pq)


def calculate_js_dis(target, approx):
    min = np.min(target)
    max = np.max(target)
    eps = 1e-8

    target_hist, target_bin_edges = np.histogram(target, bins=np.arange(min, max, (max-min)/100), density=True)
    p = (target_hist+eps) * np.diff(target_bin_edges)

    approx_hist, approx_bin_edges = np.histogram(approx, bins=np.arange(min, max, (max - min) / 100), density=True)
    q = (approx_hist+eps) * np.diff(approx_bin_edges)

    js_pq = distance.jensenshannon(p, q)
    return js_pq


