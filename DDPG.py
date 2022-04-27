# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2021年11月23日
"""
import torch
import torch.nn.functional as f
from torch import optim
from model import ActorNet, CriticNet
import numpy as np

N_VEL = 2
N_SUB = 2
state_dimension = N_VEL * N_SUB
num_actions = 2 ** (N_VEL * N_SUB)


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    a = a.detach().numpy()
    b = b.detach().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    c = torch.from_numpy(c)
    return c


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


"""
   ########################### DeepDPG ################################
"""


class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate_actor, learning_rate_critic, gamma, replacement):
        self.state_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.gamma = gamma
        self.replacement = replacement
        self.var = 1  # 动作随机探索中噪声的方差
        self.weight_dir = 'weight_Actor'

        # 初始化Actor网络

        self.actor_eval_network = ActorNet(self.state_dim, self.a_dim)
        self.actor_target_network = ActorNet(self.state_dim, self.a_dim)

        self.actor_optimizer = optim.Adam(self.actor_eval_network.parameters(), self.lr_actor)

        # 初始化Critic网络
        self.critic_eval_network = CriticNet(self.state_dim, self.a_dim)
        self.critic_target_network = CriticNet(self.state_dim, self.a_dim)

        self.critic_optimizer = optim.Adam(self.critic_eval_network.parameters(), self.lr_critic)

        hard_update(self.actor_target_network, self.actor_eval_network)  # Make sure target is with the same weight
        hard_update(self.critic_target_network, self.critic_eval_network)

    def decide_action(self, s_actor):
        # s_actor = torch.from_numpy(s_actor).float()
        s_actor = torch.unsqueeze(s_actor, 0)
        # a_actor = torch.unsqueeze(a_actor, 0)
        # 因为actor_net返回的是个只有一行的二维数组([[]])，所以要降维将其转化为一维数组([])输出
        action1 = self.actor_eval_network(s_actor).squeeze(0)
        action1_reshape = action1.reshape(N_SUB, N_VEL)

        # actions = torch.multiply(self.action_bound, x)
        # a2 = torch.mul(a1, actions)
        # a2_reshape = torch.reshape(, (-1, N_SUB, N_VEL))
        # a21 = torch.sum(a2_reshape, dim=1)
        # a21_reshape = (a21 > self.action_bound).unsqueeze(1)
        # flag = a21_reshape.repeat(1, N_SUB, 1)
        # a22 = a21.unsqueeze(1)
        # max_list = torch.clamp(a22, min=1e35)
        # a22 = torch.where(a22 == 0, max_list, a22)
        # a23 = torch.where(flag, torch.div(a2_reshape, a22), a2_reshape)
        # scaled_a = torch.reshape(a23, (-1, state_dimension))

        # 给action加上方差为var=1的高斯噪声，并使用clamp()函数把a的每个维度的值都限制在[0, P_max]之内
        action2 = torch.mul(self.action_bound, torch.clamp(torch.normal(action1_reshape, self.var), 0, 1))
        action21 = torch.sum(action2, dim=0)
        action21_reshape = (action21 > self.action_bound).unsqueeze(0)
        flag_actor = torch.repeat_interleave(action21_reshape, N_SUB, dim=0)
        action22 = action21.unsqueeze(0)
        action23 = torch.where(flag_actor, torch.mul(self.action_bound, div0(action2, action22)), action2)
        a_actor = action23.view(-1, state_dimension)
        # a_actor = torch.squeeze(action23, dim=1)

        return a_actor

    def learn(self, state_batch, action_batch, reward_batch, next_state_batch):
        action_batch = action_batch.clone().detach().requires_grad_(True)

        # 更新连接参数
        # 将网络切换到训练模式
        self.actor_eval_network.train()
        self.critic_eval_network.train()

        # 准备目标 q 批次
        next_q_values = self.critic_target_network(next_state_batch, self.actor_target_network(next_state_batch))
        next_q_values.volatile = False

        target_q_batch = reward_batch + self.gamma * next_q_values

        # Critic update
        self.critic_optimizer.zero_grad()
        q_batch = self.critic_eval_network(state_batch, action_batch)

        value_loss = f.mse_loss(q_batch, target_q_batch, reduction='mean')
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        self.actor_optimizer.zero_grad()

        policy_loss = -self.critic_eval_network(state_batch, self.actor_eval_network(state_batch))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Target update
        soft_update(self.actor_target_network, self.actor_eval_network, self.replacement['tau'])
        soft_update(self.critic_target_network, self.critic_eval_network, self.replacement['tau'])

    def load_weight_from_pkl(self):  # 此函数用于加载训练好的权重
        checkpoint = torch.load('./weight_Actor/save_para.pth')
        self.actor_eval_network.load_state_dict(checkpoint['actor_eval_network_state_dict'])
        self.actor_target_network.load_state_dict(checkpoint['actor_target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.cost_his = checkpoint['loss']

    def save_weight_to_pkl(self):
        torch.save({
            # 'epoch': epochs,
            'actor_eval_network_state_dict': self.actor_eval_network.state_dict(),
            'actor_target_network_state_dict': self.actor_target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss': self.cost_his,
        }, './weight_Actor/save_para.pth')
        print(f"Actor_network参数保存成功")

