# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 04:36:38 2020

@author: jpzxshi
"""
import os

import numpy as np

from utils import timing, TorchData
from math_ops import cdist
from data_load import load_MNIST

class CCData(TorchData):
    '''Data for studying the cover complexity.
    '''
    def __init__(self):
        super(CCData, self).__init__()
        
        self.__dists = None
        self.__deltaT = None
        self.__TC = None
        self.__SC = None
        self.__MC = None
        self.__CD = None
        self.__CC = None
    
    @property
    def dists(self):
        if self.__dists is None:
            self.__dists = cdist(self.X_test_np, self.X_train_np, metric='euclidean')
        return self.__dists
    
    @property
    def deltaT(self):
        if self.__deltaT is None:
            X_train = self.X_train_np
            y_train = self.y_train_np
            dim = X_train.shape[1]
            K = y_train.shape[1]
            dmin = np.sqrt(dim)
            for i in range(K):
                for j in range(i + 1, K):
                    mask_i = np.argmax(y_train, axis=1) == i
                    mask_j = np.argmax(y_train, axis=1) == j
                    if np.any(mask_i) and np.any(mask_j):
                        dmin = min(dmin, np.min(cdist(X_train[mask_i], X_train[mask_j])))
            self.__deltaT = dmin
        return self.__deltaT
    
    @property
    def TC(self):
        if self.__TC is None:
            self.__TC = CCData.rho(self.dists, self.dim)
        return self.__TC
    
    @property
    def SC(self):
        if self.__SC is None:
            rho_list = []
            for i in range(self.K):
                rho_list.append(CCData.rho(self.__get_label_dists(i, i), self.dim))
            self.__SC = np.mean(rho_list)
        return self.__SC

    @property
    def MC(self):
        if self.__MC is None:
            rho_list = []
            # 求解不同标签的概率值
            for i in range(self.K):
                for j in range(self.K):
                    if i != j:
                        # 遍历每个标签
                        rho_list.append(CCData.rho(self.__get_label_dists(i, j), self.dim))
            # np.mean无论是多少数据，都可以算得直接的平均值
            self.__MC = np.mean(rho_list)
        return self.__MC            
    
    @property
    def CD(self):
        if self.__CD is None:
            self.__CD = self.SC - self.MC
        return self.__CD
    
    @property
    def CC(self):
        if self.__CC is None:
            print('Computing CC...', flush=True)
            @timing
            def Computing():
                return (1 - self.TC) / self.CD
            self.__CC = Computing()
        return self.__CC
    
    
    def __get_label_dists(self, test_label, train_label):
        '''

        y_test_np是1000*10的，1000涨测试图片，每个图片【0，0，0，0，0，0，0，0，0，1】标志着是哪个数字，
        np.argmax(self.y_test_np, axis=1)表示固定一行，在行中的数据进行比较，即找到每个图片标签[0,0,0,0,0,0,0,0,0,1]中最大的下标
        '''
        # mask_test如果这一行的标签,记录每个图片是否是当前查询的这个标签，记录在长度为1000的向量中
        # 记录测试集的每个图片的标签是否和该标签一致，
        mask_test = np.argmax(self.y_test_np, axis=1) == test_label
        # 同样的对训练集的标签也做记录,每个样本对应一个值，True or False
        # 记录训练集的每个图片的标签是否和该标签一致
        mask_train = np.argmax(self.y_train_np, axis=1) == train_label
        # dist是一个记录距离的二维数组，返回第一个维度即mask_test中为true的行，返回第二个维度mask_train中为true的列
        # 相同标签，就是说测试集和训练集统一标签，
        # 不同标签，传入的test_label和train_label不一样，然后两个mask_test和mask_train也不一样，还是返回都是true的dist，这是不同标签的dist
        return self.dists[mask_test, :][:, mask_train]
    
    @staticmethod
    def rho(dists, dim, n=1000):
        '''dists: [test_n, train_n]
        '''
        # dim=784维度,
        # np.any()对矩阵所有元素做或运算，存在True则返回True
        h = lambda r: np.mean(np.any(dists < r, axis=1))
        diam = np.sqrt(dim)
        step = diam / n
        # map函数map(function, arr）arr中的所有元素做function操作
        # list(map(function, arr))返回到列表中
        # 即对所有的0到diam（最大距离）中的每一步数据做计算lambda表达式，然后做积分
        return np.sum(list(map(h, np.arange(0, diam, step)))) * step / diam
    
    


class MNIST(CCData):
    '''Dataset MNIST.
    '''
    def __init__(self):
        super(MNIST, self).__init__()
        # load data
        path = os.getcwd() + '/datasets/mnist_data/'
        mnist = load_MNIST(path, pixel_normalization=True, one_hot=True)
        
        self.X_train = mnist['X_train'][5000:5020]    #(55000, 784)
        # self.X_train = mnist['X_train'][5000:]    #(55000, 784)
        self.y_train = mnist['y_train'][5000:5020]    #(55000, 10)
        # self.y_train = mnist['y_train'][5000:]    #(55000, 10)

        # self.X_test = mnist['X_test']            #(10000, 784)
        self.X_test = mnist['X_test'][0:10]       #(10000, 784)
        # self.y_test = mnist['y_test']             #(10000, 10)
        self.y_test = mnist['y_test'][0:10]      #(10000, 10)

        

def main():
    mnist = MNIST()
    print('MNIST')
    print('X_train:', mnist.X_train_np.shape, 'y_train:', mnist.y_train_np.shape)
    print('X_test:', mnist.X_test_np.shape, 'y_test:', mnist.y_test_np.shape)
    print('CC:', mnist.CC)
    print('TC:', mnist.TC, 'SC:', mnist.SC, 'MC:', mnist.MC, 'CD:', mnist.CD)

if __name__ == '__main__':
    main()
    # a = [1,2,3,4,5,6,7,8,9]
    # print(a[5:])
    # print(a[0:5])
