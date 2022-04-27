# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2021年09月12日
"""

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from DQN_And_DDPG import DeepQNetwork, DDPG
from Environment import Environ
from Replay_memory import Memory
import argparse
import torch
import ch

ch.set_ch()
"""
#####################  hyper parameters  ####################
"""

MAX_EPISODES = 100
MAX_EP_STEPS = 6000
LR_A = 0.01    # learning rate for actor
LR_C = 0.01    # learning rate for critic
LR_Q = 0.01    # DQN的学习率
GAMMA = 0.9     # reward discount
E_GREEDY = 0.9
REPLACEMENT = dict(name='soft', tau=0.01)  # 可以选择不同的replacement策略，这里选择了soft replacement
REPLACE_TARGET_ITER = 2  # DON网络的更新频率
MEMORY_SIZE = 1000
BATCH_SIZE = 100
OUTPUT_GRAPH = False
CAPACITY = 1000

env = Environ()
n_veh = env.n_Veh  # DDPG中Actor网络的输出动作维度，大小为K x M
n_RB = env.n_RB
state_dim = env.state_dim
n_actions = env.n_actions  # DQN网络中动作空间的大小，此时每个动作的维度为1
action_dim = env.action_dim  # DDPG中Actor网络的输出动作维度，大小为K x M
action_bound = env.action_bound

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use gpu or not')
parser.add_argument('--gpu_fraction', default=(0.5, 0), help='idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
parser.add_argument('--random_seed', type=int, default=123, help='Value of random seed')
opt = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set random seed
setup_seed(opt.random_seed)
random.seed(opt.random_seed)

if opt.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")


# def calc_gpu_fraction(fraction_string):
#     idx, num = fraction_string.split('/')
#     idx, num = float(idx), float(num)
#     fraction = 1 / (num - idx + 1)
#     print(" [*] GPU : %.4f" % fraction)
#     return fraction


def div0(a, b):  # 0/0=0
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


def bit_to_array(t, n):  # 将十进制数t转化为n位的二进制数数组
    t = int(t)
    s1 = [0 for _ in range(n)]
    index = -1
    while t != 0:
        s1[index] = t % 2
        t = t >> 1
        index -= 1
    return np.array(s1).astype(np.float32)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 控制GPU资源使用的两种方法
    # （1）直接限制gpu的使用率
    print(torch.cuda.is_available())
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    deepQNetwork = DeepQNetwork(CAPACITY, n_actions, state_dim, LR_Q, GAMMA, E_GREEDY, REPLACE_TARGET_ITER, BATCH_SIZE)
    ddpg = DDPG(state_dim, action_dim, action_bound, LR_A, LR_C, GAMMA, REPLACEMENT)

    M = Memory(capacity=CAPACITY)

    # 是否输出流图
    # if OUTPUT_GRAPH:
    #     tf.summary.FileWriter('logs/', sess.graph)

    # deepQNetwork.load_weight_from_pkl()
    # actor.load_weight_from_pkl()
    # critic.load_weight_from_pkl()
    # actor.var = 0.00
    # print(deepQNetwork.main_q_network.fc1.bias)

    # 训练
    for i in range(MAX_EPISODES):
        s = env.reset()
        s = torch.from_numpy(s).type(torch.FloatTensor)
        s = torch.unsqueeze(s, 0)  # 将state_dim转换为1×4
        # state = torch.reshape(state, (-1, state_dim))
        ep_reward = 0  # 记录这一回合的总的奖励reward

        for j in range(MAX_EP_STEPS):
            a1 = deepQNetwork.decide_action(s)
            a1_to_bit = bit_to_array(a1, action_dim)
            a1_to_bit = torch.from_numpy(a1_to_bit).type(torch.FloatTensor)
            # a1_to_bit = torch.unsqueeze(a1_to_bit, 0)  # 将state_dim转换为1×4

            a2 = ddpg.decide_action(s, a1_to_bit)
            a = torch.cat([a1, a2], 0)
            s_, r = env.step(a2.detach().numpy())
            s_ = torch.from_numpy(s_).type(torch.FloatTensor)
            s_ = torch.unsqueeze(s_, 0)  # 将state_dim转换为1×4
            if 1 in torch.isnan(s_):
                print(f"改状态没有后续状态")
                continue
            r = torch.from_numpy(r.reshape(1)).type(torch.FloatTensor)
            r = torch.unsqueeze(r, 0)

            # a1_to_bit = torch.unsqueeze(a1_to_bit, 0)  # 将state_dim转换为1×4

            M.push(s, a, r, s_)
            deepQNetwork.memory.push(s, a, r, s_)

            if (deepQNetwork.memory.__len__() >= MEMORY_SIZE) and (deepQNetwork.memory.index % 5 == 0):
                ddpg.var *= 0.9998

                deepQNetwork.replay()
                ddpg.learn(deepQNetwork.state_batch, deepQNetwork.action_batch[:, 1:], deepQNetwork.reward_batch,
                           deepQNetwork.non_final_next_states)

            s = s_
            ep_reward += r

        # 每个回合结束时，打印出当前的回合数以及总的reward
        print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % ddpg.var, )

    # 测试基于DRL的资源分配策略
    r_episode = np.zeros((MAX_EPISODES, 60))
    for i in range(MAX_EPISODES):
        s = env.reset()
        s = torch.from_numpy(s).type(torch.FloatTensor)
        s = torch.unsqueeze(s, 0)  # 将state_dim转换为1×4
        r_record = []
        r_100_mean = []

        for j in range(MAX_EP_STEPS):
            a1 = deepQNetwork.decide_action(s)
            a1_to_bit = bit_to_array(a1, action_dim)
            a1_to_bit = torch.from_numpy(a1_to_bit).type(torch.FloatTensor)
            a2 = ddpg.decide_action(s, a1_to_bit)
            s_, r = env.step(a2.detach().numpy())
            s_ = torch.from_numpy(s_).type(torch.FloatTensor)
            s_ = torch.unsqueeze(s_, 0)  # 将state_dim转换为1×4
            r = torch.from_numpy(r.reshape(1)).type(torch.FloatTensor)
            r = torch.unsqueeze(r, 0)
            r_record.append(r)
            if j % 100 == 0:
                r_100_mean.append(np.mean(r_record[-100:]))

            s = s_
        r_episode[i] = np.array(r_100_mean)

    print('Average rewards using DDPG:', np.mean(r_episode))
    r_episode_mean_DDPG = np.reshape(np.mean(r_episode, axis=0), -1)
    plt.plot(100 * np.arange(len(r_episode_mean_DDPG)), r_episode_mean_DDPG)
    plt.xlabel('时隙(TS)')
    plt.ylabel('上行链路NOMA系统总和能量效率(bit/J)')
    plt.title('基于DRL的资源分配策略')
    plt.show()

    # 测试随机资源分配策略
    r_episode = np.zeros((MAX_EPISODES, 60))
    for i in range(MAX_EPISODES):
        _ = env.reset()
        ep_reward = 0  # 记录这一回合的总的奖励reward
        r_record = []
        r_100_mean = []

        for j in range(MAX_EP_STEPS):
            a = action_bound * np.random.random(state_dim)
            a_reshape = a.reshape((n_RB, n_veh))
            a_sum = np.sum(a_reshape, axis=0)
            a_sum_reshape = (a_sum > action_bound)[np.newaxis, :]
            flag = np.repeat(a_sum_reshape, n_RB, axis=0)
            a_sum_expand = a_sum[np.newaxis, :]
            action = np.where(flag, (action_bound * div0(a_reshape, a_sum_expand)), a_reshape)

            _, r = env.step(action)
            r_record.append(r)
            if j % 100 == 0:
                r_100_mean.append(np.mean(r_record[-100:]))
            ep_reward += r

        r_episode[i] = np.array(r_100_mean)

    print('Average rewards using random selection:', np.mean(r_episode))
    r_episode_mean_random = np.reshape(np.mean(r_episode, axis=0), -1)
    plt.plot(100 * np.arange(len(r_episode_mean_random)), r_episode_mean_random)
    plt.xlabel('时隙(TS)')
    plt.ylabel('上行链路NOMA系统总和能量效率(bit/J)')
    plt.title('随机资源分配策略')
    plt.show()

    # 功率最大化资源分配策略
    r_episode = np.zeros((MAX_EPISODES, 60))
    for i in range(MAX_EPISODES):
        _ = env.reset()
        ep_reward = 0  # 记录这一回合的总的奖励reward
        r_record = []
        r_100_mean = []

        for j in range(MAX_EP_STEPS):
            a = action_bound * np.random.random(state_dim)
            a_reshape = a.reshape((n_RB, n_veh))
            a_sum = np.sum(a_reshape, axis=0)
            action = action_bound * div0(a_reshape, a_sum)

            _, r = env.step(action)
            r_record.append(r)
            if j % 100 == 0:
                r_100_mean.append(np.mean(r_record[-100:]))
            ep_reward += r

        r_episode[i] = np.array(r_100_mean)

    print('Average rewards using max strategy:', np.mean(r_episode))
    r_episode_mean_max = np.reshape(np.mean(r_episode, axis=0), -1)
    plt.plot(100 * np.arange(len(r_episode_mean_max)), r_episode_mean_max)
    plt.xlabel('时隙(TS)')
    plt.ylabel('上行链路NOMA系统总和能量效率(bit/J)')
    plt.title('功率最大化资源分配策略')
    plt.show()

    # 同时画出三种策略进行对比
    plt.plot(100 * np.arange(len(r_episode_mean_DDPG)), r_episode_mean_DDPG)
    plt.plot(100 * np.arange(len(r_episode_mean_random)), r_episode_mean_random)
    plt.plot(100 * np.arange(len(r_episode_mean_max)), r_episode_mean_max)
    plt.xlabel('时隙(TS)')
    plt.ylabel('上行链路NOMA系统总和能量效率(bit/J)')
    plt.legend(["基于DRL的资源分配策略", "随机资源分配策略", "功率最大化资源分配策略"])
    plt.show()

    deepQNetwork.save_weight_to_pkl()
    ddpg.save_weight_to_pkl()
