#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:44:59 2021

@author: ckjensen
"""

import os
import time

import matplotlib.pyplot as plt
import imageio

from scan_matching import do_example


SAVE_FIG = True         # Set to True to save figures
FIG_FORMAT = 'png'      # Used for the output format when saving figures
FIG_DIR = 'results'     # Directory in which to save the figures
XLIM = [-0.5, 12.5]
YLIM = [-0.5, 12.5]


def setRCParams():
    # Run this to make sure that the matplotlib plots have the correct font type
    # for an IEEE publication. Also sets font sizes and line widths for easier
    # viewing.
    plt.rcParams.update({
                'font.size': 40,
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
                'xtick.labelsize': 40,
                'ytick.labelsize': 40,
                'lines.linewidth': 4,
                'lines.markersize': 18,
                'figure.figsize': [13.333, 10]
                })
    # plt.tight_layout()


def resetRCParams():
    # Reset the matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)


if __name__ == '__main__':
    setRCParams()

    print('Running ICP...')
    tlast = time.time()
    Q, P_values, chi_values, corr_values, x_values = do_example()
    print(f'Done in {time.time() - tlast:.3} seconds.')

    plt.close('all')
    fig, ax = plt.subplots()
    _ = ax.plot(Q[0, :], Q[1, :], 'b.', label='Original (Q)')
    line = ax.plot(P_values[0][0, :], P_values[0][1, :], 'r.', label='Translated + Noise (P)')[0]

    ax.set_xlim([0, 375])
    ax.set_ylim([0, 375])
    ax.axis('equal')
    ax.set_title('Scan Matching Using Least Squares ICP')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    fig.savefig(os.path.join('..', FIG_DIR, f'ICP_0.{FIG_FORMAT}'), format=FIG_FORMAT)
    for i, P in enumerate(P_values[1:]):
        line.set_data(P)
        fig.savefig(os.path.join('..', FIG_DIR, f'ICP_{i+1}.{FIG_FORMAT}'), format=FIG_FORMAT)

    resetRCParams()

    print('Creating ICP gif...')
    with imageio.get_writer('ICP_Example2.gif', mode='I') as w:
        for i in range(64):
            file = os.path.join('..', FIG_DIR, f'ICP_{i}.{FIG_FORMAT}')
            w.append_data(imageio.imread(file))
    print('Done')
