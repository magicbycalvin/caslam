#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:24:35 2021

@author: ckjensen
"""

import os

import cv2
import matplotlib.pyplot as plt
from numba import njit
import numpy as np


def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]]).squeeze()


def dR(theta):
    return np.array([[-np.sin(theta), -np.cos(theta)],
                     [np.cos(theta), -np.sin(theta)]]).squeeze()


def jacobian(x, Pn):
    theta = x[2]
    J = np.zeros((2, 3))
    J[:2, :2] = np.eye(2)
    J[:2, 2] = (dR(theta)@Pn).squeeze()

    return J


def error(x, Pn, Qn):
    theta = x[2]
    return R(theta)@Pn + x[:2] - Qn


def initialize(x, P, Q, corr, kernel=lambda dist: 1.0):
    H = np.zeros((3, 3))
    g = np.zeros((3, 1))
    chi = 0

    for i, j in corr:
        Pn = P[:, i, np.newaxis]
        Qn = Q[:, j, np.newaxis]
        e = error(x, Pn, Qn)
        weight = kernel(e)
        J = jacobian(x, Pn)
        H += weight*J.T@J
        g += weight*J.T@e
        chi += e.T@e

    return H, g, chi


@njit(cache=True)
def get_correspondences(P, Q):
    corr = []

    for i, Pn in enumerate(P.T):
        dmin = np.inf
        idx = -1

        for j, Qn in enumerate(Q.T):
            # dist = np.linalg.norm(Qn-Pn)
            dist = np.sum((Qn-Pn)**2)
            if dist < dmin:
                dmin = dist
                idx = j

        corr.append((i, idx))

    return corr


def icp_least_squares(P, Q, niter=50, kernel=lambda distance: 1.0, print_iter=False):
    x = np.zeros((3, 1))
    corr_values = []
    chi_values = []
    x_values = []
    P_values = [P.copy()]
    P_xform = P.copy()

    for i in range(niter):
        if print_iter: print(f'Iteration number: {i}')

        corr = get_correspondences(P_xform, Q)
        corr_values.append(corr)

        H, g, chi = initialize(x, P, Q, corr, kernel)
        dx = np.linalg.lstsq(H, -g, rcond=None)[0]
        x += dx
        x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2]))
        chi_values.append(chi[0])
        x_values.append(x.copy())

        P_xform = R(x[2])@P + x[:2]
        P_values.append(P_xform)

        if print_iter:
            print(f'[+] Chi^2 value: {chi}')

    corr_values.append(corr_values[-1])

    return P_values, chi_values, corr_values


if __name__ == '__main__':
    fname = os.path.join('..', 'data', 'images', 'doge.png')
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    theta_true = np.pi/4
    t_true = np.array([[200.],
                       [100.]])
    R_true = R(theta_true)

    ytmp, xtmp = np.where(img==0)
    np.random.seed(5)
    subsample = np.random.choice(range(len(xtmp)), size=int(0.5*len(xtmp)), replace=False)
    Q = np.vstack([xtmp[subsample], img.shape[0] - ytmp[subsample]]).astype(float)
    P = R_true@Q + t_true + np.random.randn(*Q.shape)

    P_values, chi_values, corr_values = icp_least_squares(P, Q, niter=100, print_iter=True)

    plt.close('all')
    plt.figure()
    plt.scatter(Q[0, :], Q[1, :], label='Original (Q)')
    plt.scatter(P[0, :], P[1, :], label='Translated + Noise (P)')
    plt.scatter(P_values[-1][0, :], P_values[-1][1, :], label='Result')
    plt.xlim([0, 375])
    plt.ylim([0, 375])
    plt.axis('equal')
    plt.title('Original and Translated Images')
    plt.legend()