# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2021年11月29日
"""
import random

import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_next'))


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity  # 下面memory的最大长度
        self.memory = []  # 存储过往经验
        self.index = 0  # 表示要保存的索引

    def push(self, state, action, reward, state_next):
        # 将transition = (state, action, reward, state_next)保存在存储器中

        if self.__len__() < self.capacity:
            self.memory.append(None)  # 内存未满时添加

        # 使用namedtuple对象Transition将值和字段名称保存为一对
        self.memory[self.index] = Transition(state, action, reward, state_next)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        assert self.__len__() >= self.capacity, 'Memory has not been fulfilled'
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # 返回当前memory的长度
        return len(self.memory)
