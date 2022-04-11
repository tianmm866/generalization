# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 02:37:43 2019

@author: jpzxshi
"""
import os
import itertools

import numpy as np
import torch


#
# numpy
#
def inverse_modulus_continuity(f, dim, eps, Nx=1000, save_iters=False):
    """Compute the inverse of the modulus of continuity.
    The domain of 'f' is [0, 1]^{dim}.
    """
    if dim == 1:
        X = np.linspace(0, 1, num=Nx)[:, None]
        Y = f(X)
        dx = X[1, 0] - X[0, 0]
        k = 0
        while k < Nx:
            if np.any(np.linalg.norm(Y[:Nx-k] - Y[k:], ord=np.inf, axis=1) >= eps):
                return k * dx
            k += 1
        return 1
    elif dim == 2:
        iters = list(itertools.product(range(Nx), range(Nx)))
        X = np.array(iters) / (Nx - 1)
        Y = f(X)
        data_iters = 'iters_{}.npy'.format(Y.shape[0])
        if save_iters and os.path.isfile(data_iters):
            iters = np.load(data_iters)
        else:
            iters.sort(key=lambda point: np.linalg.norm(point))
            if save_iters:  
                np.save(data_iters, iters)
                print('save', data_iters)
        Ymat = Y.reshape([Nx, Nx, Y.shape[-1]])
        def runover(m, n):
            dY = np.linalg.norm(Ymat[:Nx - m, :Nx - n] - Ymat[m:, n:], ord=np.inf, axis=2)
            if np.any(dY >= eps):
                return np.linalg.norm([m, n]) / (Nx - 1)
            return None
        for it in iters:
            delta = runover(*it)
            if delta is not None: return delta
        return np.sqrt(dim)
    else: return None
    

def cdist(x, y, metric='euclidean'):
    '''
    X是测试集，y是训练集
    num_1：平方项，测试集的数量，num_2训练集的数量,
    点乘扩展成一个num_1(测试集长度)xnum2（训练集长度）的矩阵，其中，每一行的元素均相同，都是图片像素点的平方和

    dist_2 ：平方项，构造一个[测试集长度，训练集长度]的真实矩阵，其中每行的值对应一个训练集图片的像素平方和

    dist_3：-2ab选项，构造一个[测试集长度，训练集长度]的真实矩阵，
    np.dot为点积，内积，向量做内积，矩阵做矩阵乘法
    这里需要注意的是一维矩阵和一维向量的区别，一维向量的shape是(5, ), 而一维矩阵的shape是(5, 1), 若两个参数a和b都是一维向量则是计算的点积，但是当其中有一个是矩阵时（包括一维矩阵），dot便进行矩阵乘法运算，同时若有个参数为向量，会自动转换为一维矩阵进行计算。
    测试集和训练集做矩阵相乘【1000，784】X[784,200]=【1000，200】

    dists:最后结果，平方的结果，返回训练集和测试集之间的距离
    '''
    dists = None
    if metric == 'euclidean':
        # 压缩列，就是将每个图片的784个像素点的值都加到一起，keepdim = True保持维度不变
        num_1, num_2 = x.shape[0], y.shape[0]
        dist_1 = np.sum(np.square(x), axis=1, keepdims=True) * np.ones(num_2)
        dist_2 = np.ones([num_1, 1]) * np.sum(np.square(y), axis=1)
        dist_3 = - 2 * np.dot(x, y.transpose())
        dists = np.sqrt(np.abs(dist_1 + dist_2 + dist_3))
    else: raise NotImplementedError
    return dists

#
# torch
#
def Softmax(X, dim):
    e_x = torch.exp(X - torch.max(X, dim=dim, keepdim=True)[0])
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)
    
def Cross_entropy(p, q):
    if p is not None and q is not None:
        return torch.mean(-torch.sum(p * torch.log(q), dim=1))
    return torch.FloatTensor([-1])

def Test_accuracy(data, net):
    y_pred = net(data.X_test)
    diff = torch.argmax(y_pred, dim=1) == torch.argmax(data.y_test, dim=1)
    return torch.mean(diff.type(torch.FloatTensor)).item()
         
def Grad(y, x):
    '''
    y: [N, Ny] or [Ny]
    x: [N, Nx] or [Nx]
    Return dy/dx ([N, Ny, Nx] or [Ny, Nx]).
    '''
    N = y.size(0) if len(y.size()) == 2 else 1
    Ny = y.size()[-1]
    Nx = x.size()[-1]
    z = torch.ones([N])
    if x.is_cuda:
        z=z.cuda()
    dy = []
    if len(y.size()) == 2:
        for i in range(Ny):
            dy.append(torch.autograd.grad(y[:, i], x, grad_outputs=z, create_graph=True)[0])
        res = torch.cat(dy, 1).view([N, Ny, Nx])
    else:
        for i in range(Ny):
            dy.append(torch.autograd.grad(y[i], x, grad_outputs=z, create_graph=True)[0])
        res = torch.cat(dy, 0).view([Ny, Nx])
    return res

     

   
def main():
    f = lambda x: x @ np.array([[1], [1]])
    imc = inverse_modulus_continuity(f, dim=2, eps=0.5, Nx=200, save_iters=False)
    print(imc)
    

if __name__ == '__main__':
    main()
        