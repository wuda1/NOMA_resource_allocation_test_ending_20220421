# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2021年09月12日
"""

import numpy as np
import random
import math


class V2IChannels:
    """Simulator of the V2I channels (V2I信道模拟器)"""

    def __init__(self, n_veh, n_rb):
        self.n_Veh = n_veh
        self.n_RB = n_rb
        self.distances = None
        self.PathLoss = None
        self.FastFading = None

    def update_distances(self, distances):
        self.distances = distances

    def update_path_loss(self):
        # 用于更新V2I之间的路径损失PL
        self.PathLoss = np.zeros(len(self.distances))
        for i in range(len(self.distances)):
            self.PathLoss[i] = 122.0 + 38.0 * np.log10(self.distances[i] / 1000)

    """
        初始化快（小规模）衰落的信道特征h(dB)，self.FastFading为一个二维数组，
        第二维是资源块的数量，即V2I link的每个子频带，都有着它自己的信道特性h(self.FastFading)
        二维数组的每一个元素都是实部和虚部都服从正太分布的一个复信特性
    """
    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size=(self.n_RB, self.n_Veh)) +
                            1j * np.random.normal(size=(self.n_RB, self.n_Veh)))
        self.FastFading = 20 * np.log10(np.abs(h))


class Vehicle:
    """
        Vehicle simulator: include all the information for a vehicle
        车辆类，每一辆车的信息包括：位置、方向、速度、它到的3个neighbors的距离、目的地
    """
    def __init__(self, start_distance, velocity):
        self.distance = start_distance  # 每个位置信号包含着（x, y）坐标的两个变量
        self.velocity = velocity  # 1m/s


class Environ:
    """
        环境模拟器：（1）为agent提供状态s和反馈reward
        （2）根据agent所采取的动作action，环境会返回更新的状态s（t+1）
    """

    # 初始化环境参数
    def __init__(self):
        self.time_step = 0.1  # 更新车辆位置信息的时间间隔
        self.vehicles = []  # 用于存储摆放在环境中的车辆对象（有Vehicle类生成）
        self.PO = 5
        self.sig2_dB = -174  # 环境高斯噪声功率（dBm）
        self.sig2 = 10**(self.sig2_dB/10)/1000  # 环境高斯噪声功率（W）
        self.delta_distance = []
        self.n_RB = 2  # 子频带的数量
        self.n_Veh = 2  # 车辆的数量
        self.state_dim = self.n_RB * self.n_Veh
        self.n_actions = 2 ** (self.n_RB * self.n_Veh)
        self.action_dim = self.n_RB * self.n_Veh
        self.action_bound = 1.0
        self.V2IChannels = V2IChannels(self.n_Veh, self.n_RB)  # 创建V2I信道对象
        self.reset_pointer = 0
        self.radius = 1000
        self.start_velocity = 1  # 用户的初始速度，此处为1m/s
        self.V2I_channels_abs = None
        self.V2I_channels_with_fast_fading = None

    # 在一个圆内随机取若干个坐标点
    @staticmethod
    def get_distance(num, radius, center_x=0, center_y=0):
        distance = []
        for i in range(num):
            while True:
                x = random.uniform(-radius, radius)
                y = random.uniform(-radius, radius)
                if (x ** 2) + (y ** 2) <= (radius ** 2):
                    distance.append(math.hypot((int(x)+center_x), (int(y)+center_y)))
                    break
        return distance

    # 用于添加n个新的车辆对象
    def add_new_vehicles_by_number(self, n, start_velocity):
        start_distance = self.get_distance(n, self.radius)
        for i in range(n):
            self.vehicles.append(Vehicle(start_distance[i], start_velocity))

    def renew_positions(self):
        """
        This function update the position of each vehicle
        这个函数用于更新每一辆车的位置信息，每0.1s更新一次
        :return:
        """
        for i in range(len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_step * np.random.uniform(-1.0, 1.0)
            self.vehicles[i].distance = np.clip(self.vehicles[i].distance + delta_distance, 0, self.radius)

    def renew_channel(self):
        """
        This function updates all the channels including V2I channels
        更新V2I信道大尺度衰落的信道特性参数
        :return:
        """
        distances = [c.distance for c in self.vehicles]
        self.V2IChannels.update_distances(distances)
        self.V2IChannels.update_path_loss()

        # 获得新的大尺度衰落（dB）
        self.V2I_channels_abs = self.V2IChannels.PathLoss   # 一维

    def renew_channels_fast_fading(self):
        """
        This function updates all the channels including V2I channels
        更新所有的信道包括V2I信道
        :return:
        """
        # 先更新所有信道的大尺度衰落参数
        self.renew_channel()
        # 更新小尺度衰落时的信道特性h（dB）
        self.V2IChannels.update_fast_fading()

        """
        repeat(a, repeats, axis=None)，其中a为输入的数组，repeats为a中每个元素重复的次数
        axis代表重复数值的方向，axis=0代表y轴方向，axis=1代表x轴方向，axis=2代表z轴方向
        将V2I_channels_abs转化为二维，第二维为子频带数量。此时同一个V2I link的不同子频带的self.V2I_channels_abs值相同
        """
        v2i_channels_with_fast_fading = np.repeat(self.V2I_channels_abs[np.newaxis, :], self.n_RB, axis=0)
        # 计算出同时具有大尺度和小尺度衰落时的V2I信道特性（h（dB）），即Gk，m（t）
        self.V2I_channels_with_fast_fading = np.sqrt(v2i_channels_with_fast_fading) * self.V2IChannels.FastFading

    def compute_reward(self, action):
        """
        Used for Training
        add the power dimension to the action selection
        :param action:
        :return:
        """
        power_selection = np.array(action.copy()).reshape((self.n_RB, self.n_Veh))
        power_selection_db = np.zeros((self.n_RB, self.n_Veh))
        for k in range(self.n_RB):
            for m in range(self.n_Veh):
                if power_selection[k, m] != 0:
                    power_selection_db[k, m] = 10 * np.log10(power_selection[k, m])  # 功率选择

        interference = np.zeros((self.n_RB, self.n_Veh))
        v2i_rate_list = np.zeros((self.n_RB, self.n_Veh))

        for k in range(self.n_RB):
            for m in range(self.n_Veh):
                if power_selection[k, m] != 0:
                    for i in range(self.n_Veh):
                        if i != m:
                            interference[k, m] += 10 ** ((power_selection_db[k, i] *
                                                          self.V2I_channels_with_fast_fading[k, m] ** 2) / 10)
                    interference[k, m] += self.sig2

        for k in range(self.n_RB):
            for m in range(self.n_Veh):
                if power_selection[k, m] != 0:
                    v2i_rate_list[k, m] = np.log2(1 + np.divide(10 ** ((power_selection_db[k, m] *
                                                                self.V2I_channels_with_fast_fading[k, m] ** 2) / 10),
                                                                interference[k, m]))

        ee = np.divide(v2i_rate_list, (power_selection + self.PO))

        return np.sum(ee)

    def step(self, action):
        """
        这个函数用于计算在训练时采用了动作action之后，环境的反馈reward
        :param action:
        :return:
        """
        action_temp = action.copy()
        reward_sum = self.compute_reward(action_temp)  # 计算出采取了action之后各个信道的容量

        # 每采取一个action，更新一次环境
        self.renew_positions()
        self.renew_channels_fast_fading()

        s_ = self.V2I_channels_with_fast_fading.reshape((-1))

        return s_, reward_sum

    def reset(self):
        self.vehicles = []
        self.add_new_vehicles_by_number(self.n_Veh, self.start_velocity)
        self.renew_channels_fast_fading()
        return self.V2I_channels_with_fast_fading.reshape((-1))


def bit_to_array(t, n):
    t = int(t)
    s = [0 for i in range(n)]
    i = -1
    while t != 0:
        s[i] = t % 2
        t = t >> 1
        i -= 1
    return np.array(s)




