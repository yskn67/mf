#!/uer/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np


class MatrixFactorization():

    def __init__(self, U=None, V=None, K=64, epochs=500, reg=0.01, alpha=0.5, threshold=0.001, debug=False):
        self.U = U
        self.V = V
        self.K = K
        self.epochs = epochs
        self.reg = reg
        self.alpha = alpha
        self.threshold = threshold
        self.debug = debug

    def init_factors(self, n, k, mean=0.0, std=0.01):
        if self.debug:
            return np.zeros((n, k)) + np.float64(0.5)
        else:
            return np.random.normal(mean, std, (n, k))

    def get_error(self, mat):
        error = self.reg * ((self.U ** 2).sum() + (self.V ** 2).sum())
        for i in range(self.U_num):
            l = 0.0
            for j in range(self.V_num):
                if mat[i, j] != 0.0:
                    pred = np.dot(self.U[i, :], self.V.T[:, j])
                    l += self.W[i, j] * pow(mat[i, j] - pred, 2)
                    l -= self.C[j] * pow(pred, 2)
            l += np.dot(np.dot(self.SV, self.U[i, :]), self.U[i, :])
            error += l
        return error

    def initW(self, mat):
        W = np.zeros(mat.shape)
        W[np.where(mat > 0)] = 1
        self.W = W

    def initC(self, mat, c0=1.0):
        c = np.zeros(self.V_num, dtype=np.float64)
        sum = np.float64(0)
        Z = np.float64(0)
        for i in range(self.V_num):
            c[i] = mat[:, i].sum()
        sum = c.sum()
        c = pow(c / sum, self.alpha)
        Z = c.sum()
        C = c0 * c / Z
        self.C = C

    def initS(self):
        SU = np.dot(self.U.T, self.U)
        SV = np.zeros((self.K, self.K), dtype=np.float64)
        for k in range(self.K):
            for f in range(k + 1):
                val = 0.0
                for j in range(self.V.shape[0]):
                    val += self.C[j] * self.V[j, k] * self.V[j, f]
                SV[k, f] = val
                SV[f, k] = val
        self.SU = SU
        self.SV = SV

    def update_user(self, mat, i):
        item_idx = np.where(mat[i, :] == 1.0)[0]
        if len(item_idx) == 0:
            return
        for j in item_idx:
            self.pred_items[j] = np.dot(self.U[i, :], self.V.T[:, j])
            self.rating_items[j] = mat[i, j]
            self.w_items[j] = self.W[i, j]
        oldUvec = deepcopy(self.U[i, :])
        for k in range(self.K):
            denom = 0.0
            numer = -1 * np.dot(self.U[i, :], self.SV[k, :].T)
            numer += self.U[i, k] * self.SV[k, k]
            for j in item_idx:
                self.pred_items[j] -= self.U[i, k] * self.V[j, k]
                numer += (self.w_items[j] * self.rating_items[j] - (self.w_items[j] - self.C[j]) * self.pred_items[j]) * self.V[j, k]
                denom += (self.w_items[j] - self.C[j]) * self.V[j, k] * self.V[j, k]
            denom += self.SV[k, k] + self.reg
            self.U[i, k] = numer / denom
            for j in item_idx:
                self.pred_items[j] += self.U[i, k] * self.V[j, k]
        for k in range(self.K):
            for f in range(k + 1):
                val = self.SU[k, f] - (oldUvec[k] * oldUvec[f]) + (self.U[i, k] * self.U[i, f])
                self.SU[k, f] = val
                self.SU[f, k] = val

    def update_item(self, mat, j):
        user_idx = np.where(mat[:, j] == 1.0)[0]
        if len(user_idx) == 0:
            return
        for i in user_idx:
            self.pred_users[i] = np.dot(self.U[i, :], self.V.T[:, j])
            self.rating_users[i] = mat[i, j]
            self.w_users[i] = self.W[i, j]
        oldVvec = deepcopy(self.V[j, :])
        for k in range(self.K):
            denom = 0.0
            numer = -1 * np.dot(self.V[j, :], self.SU[k, :].T)
            numer += self.V[j, k] * self.SU[k, k]
            numer *= self.C[j]
            for i in user_idx:
                self.pred_users[i] -= self.U[i, k] * self.V[j, k]
                numer += (self.w_users[i] * self.rating_users[i] - (self.w_users[i] - self.C[j]) * self.pred_users[i]) * self.U[i, k]
                denom += (self.w_users[i] - self.C[j]) * self.U[i, k] * self.U[i, k]
            denom += self.C[j] * self.SU[k, k] + self.reg
            self.V[j, k] = numer / denom
            for i in user_idx:
                self.pred_users[i] += self.U[i, k] * self.V[j, k]
        for k in range(self.K):
            for f in range(k + 1):
                val = self.SV[k, f] - (oldVvec[k] * oldVvec[f] * self.C[j]) + (self.V[j, k] * self.V[j, f] * self.C[j])
                self.SV[k, f] = val
                self.SV[f, k] = val

    def fit(self, mat):
        self.U_num, self.V_num = mat.shape
        if self.U is None or self.V is None or self.debug:
            self.U = self.init_factors(self.U_num, self.K)
            self.V = self.init_factors(self.V_num, self.K)
        self.initW(mat)
        self.initC(mat)
        self.initS()

        # cache
        self.pred_users = np.zeros(self.U_num, dtype=np.float64)
        self.pred_items = np.zeros(self.V_num, dtype=np.float64)
        self.rating_users = np.zeros(self.U_num, dtype=np.float64)
        self.rating_items = np.zeros(self.V_num, dtype=np.float64)
        self.w_users = np.zeros(self.U_num, dtype=np.float64)
        self.w_items = np.zeros(self.V_num, dtype=np.float64)

        for epoch in range(self.epochs):
            for i in range(self.U_num):
                self.update_user(mat, i)
            for j in range(self.V_num):
                self.update_item(mat, j)
            error = self.get_error(mat)
            if self.debug:
                print('error:', error)


if __name__ == '__main__':
    mat = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    mf = MatrixFactorization(K=2, epochs=30, alpha=0.75, debug=True)
    mf.fit(mat)
    print(mat)
    print(np.dot(mf.U, mf.V.T))
    print('last error:', mf.get_error(mat))
