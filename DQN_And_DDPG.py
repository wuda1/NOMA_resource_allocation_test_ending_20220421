# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2021年11月23日
"""
import torch
import torch.nn.functional as f
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from Replay_memory import Memory
from Replay_memory import Transition
from model import Net, ActorNet, CriticNet

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


class DeepQNetwork:
    def __init__(self, capacity, n_actions, n_states, learning_rate, reward_decay,
                 e_greedy, replace_target_iter, batch_size, e_greedy_increment=None):

        # 创建存储经验的经验池
        self.memory = Memory(capacity)
        self.batch = None
        self.state_batch = None
        self.action_batch = None
        self.reward_batch = None
        self.non_final_next_states = None
        self.action_next_batch = None
        self.expected_state_action_values = None
        self.state_action_values = None

        self.n_states = n_states   # self.n_states = M
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy  # epsilon 的最大值
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon_increment = e_greedy_increment  # epsilon 的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 是否开启探索模式, 并逐步减少探索次数

        # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 1  # 每5步学习一次，每10步更新一次target network的参数

        # 创建 [target_net, evaluate_net]
        self.counter = state_dimension - 1
        self.main_q_network = Net(n_states, n_actions)
        self.target_q_network = Net(n_states, n_actions)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), self.lr)

        # 记录每次训练时的损失, 用于最后plot出来观看
        self.cost_his = []

        self.weight_dir = 'weight_DQN'

    def replay(self):
        """
        经验回放学习网络的连接参数
        :return:None
        """
        # # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_target_q_network()

        # 1.检查经验池大小
        if self.memory.__len__() < self.batch_size:
            return

        # 2.创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states, self.action_next_batch \
            = self.make_mini_batch()

        # 3. 找到Q(s_t, a_t)作为监督信息
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4.更新参数
        self.update_main_q_network()

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1

    def make_mini_batch(self):
        """
        2.创建小批量数据
        :return:
        """

        # 2.1 从经验池中获取小批量数据
        transitions = self.memory.sample(self.batch_size)

        # 2.2 将每个变量转换为与小批量数据对应的形式
        # transitions表示1步的(state, action, state_next, reward)对于BATCH_SIZE个
        # 即(state, action, state_next, reward)×BATCH_SIZE
        # 它变成小批量数据，即
        # 设为(state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 2.3 将每个变量的元素转换为与小批量数据对应的形式
        # 例如，state原本为BATCH_SIZE个[torch.FloatTensor of size 1x4]
        # 将其转换为[torch.FloatTensor of size BATCH_SIZEx4]
        # cat是Concatenates（连接）
        state_batch = torch.cat(batch.state)
        # state_batch = state_batch.view(state_dimension, -1)
        action_batch = torch.cat(batch.action)
        action_batch = action_batch.view(-1, 5)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.state_next
                                           if s is not None])
        action_next_batch = torch.cat(batch.action_next)
        action_next_batch = action_next_batch.view(-1, 5)

        return batch, state_batch, action_batch, reward_batch, non_final_next_states, action_next_batch

    def get_expected_state_action_values(self):
        """
        3.求作为教师信号的Q(s t, a t)值
        :return:
        """

        # 3.1将网络切换为推理模式
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2求网络输出的Q(s_t, a_t)
        # self.model(state_batch)输出左右两个Q值
        # [torch.FloatTensor of size BATCH_SIZEx2]变成了。
        # 为了求出与从这里执行的动作a_t对应的Q值，求出在action_batch中进行的行动a_t是向左还是向右的index
        # 用gather抽出与之对应的Q值。
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch[:, :1].long())

        # 3.3求max{Q(s_t+1, a)}值。但要注意是否有以下状态。

        # 创建索引掩码，检查cartpole是否为done，是否存在next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.state_next))).bool()
        # 首先将一切设为0
        next_state_values = torch.zeros(self.batch_size)

        a_m = torch.zeros(self.batch_size).type(torch.LongTensor)
        # 从Main Q- network求出下一状态下最大Q值的行为a_m
        # 最后[1]回复对应行动的index
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]

        # 只过滤为存在以下状态的物体，将size 32变为32×1
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 从target Q- network求出有下列状态的index的行为a_m的Q值
        # 用detach()取出
        # 用squeeze()将size[minibatch×1]改为[minibatch]。
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = self.reward_batch + self.gamma * next_state_values.view(-1, 1)
        return expected_state_action_values

    def decide_action(self, state):

        # self.epsilon = 0.5 * (1 / (self.epsilon + 1))

        if self.epsilon > np.random.uniform():
            self.main_q_network.eval()  # 将网络切换到推理模式
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1]
            # 获取网络输出最大值的索引 index =max(1)[1]
            # .view(1, 1)将[torch.LongTensor of size 1]转换为size 1*1大小
        else:
            # 随机返回动作
            action = torch.LongTensor([np.random.randint(0, self.n_actions)])
        return action

    def update_main_q_network(self):
        # 更新连接参数
        # 将网络切换到训练模式
        self.main_q_network.train()

        # 计算损失函数
        # expected_state_action_values是
        # size是[mini_batch]，所以un_squeeze到[mini_batch * 1]
        loss = f.mse_loss(self.state_action_values, self.expected_state_action_values, reduction='mean')
        self.cost_his.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        """
        DeepDQNで追加
        Target Q-NetworkをMainと同じにする
        :return:
        """
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.xlabel('Training step')
        plt.ylabel('loss')
        plt.show()

    def binary(self, q):
        self.counter = state_dimension - 1
        t = torch.unsqueeze(q.max(1)[1], 1)
        temp = torch.zeros_like(t)
        temp_extend = temp.repeat(1, state_dimension)
        n = temp
        while torch.logical_not(torch.all(torch.eq(t, n))):
            t1 = t % 2
            t2 = t >> 1
            numcolunms = temp_extend.shape[1]
            new_tensor_left = temp_extend[:, :self.counter]
            new_tensor_right = temp_extend[:, self.counter+1:numcolunms]
            temp_extend = torch.cat([new_tensor_left, t1, new_tensor_right], 1)
            self.counter -= 1
        a1 = temp_extend
        a1 = a1.float()
        return a1

    def load_weight_from_pkl(self):  # 此函数用于加载训练好的权重
        checkpoint = torch.load('./weight_DQN/save_para.pth')
        self.main_q_network.load_state_dict(checkpoint['main_q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cost_his = checkpoint['loss']

    def save_weight_to_pkl(self):
        torch.save({
            # 'epoch': epochs,
            'main_q_network_state_dict': self.main_q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.cost_his,
        }, './weight_DQN/save_para.pth')
        print(f"DQN_network参数保存成功")



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

