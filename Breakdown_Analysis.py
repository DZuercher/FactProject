import os
import math
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


def make_template():
    template = np.append(
        np.zeros(10),
        sipm_vs_t(2e9, 200, 0)[0:30],
    )
    template /= template.sum()
    return template

# In[3]:

def sipm_vs_t(f_sample, N, t_offset):
    """
    The FACT SIPM pulse amplitude vs time
    """
    t = np.linspace(0, N/f_sample, num=N, endpoint=False)

    # time line in ns
    t = 1e9*(t-t_offset)

    # evaluate template, here the equation is for nano seconds
    s_vs_t = 1.626*(1.0-np.exp(-0.3803*t))*np.exp(-0.0649*t)

    # Since this is only a polynomial approx.
    # We truncate artefacts like negative amplitudes
    s_vs_t[s_vs_t < 0.0] = 0.0
    return s_vs_t


diode = 1
point_num = 4
debug_plotting = False
path = f"data/BV_{diode}_{point_num}.npy"
data = np.load(path)

template = make_template()
integrals = []
for event_id, cleandata in tqdm(enumerate(data[:, 20:-20])):
    condata = np.convolve(template, cleandata, mode='same')

    dcondata = np.convolve(
        np.diff(condata),
        np.ones(10) / 10.,
        'same'
    )

    diff_is_large_and_values_far_apart = []
    for large_derivative_pos in np.where(dcondata[25:-50] > 0.4)[0] + 25:
        i = large_derivative_pos
        _min = condata[i-25:i].min()
        _max = condata[i:i+50].max()
        if abs(_max - _min) < 5:
            continue
        diff_is_large_and_values_far_apart.append(large_derivative_pos)

    diff_is_large_and_values_far_apart = np.array(diff_is_large_and_values_far_apart, dtype=int)
    if len(diff_is_large_and_values_far_apart) == 0:
        continue

    big_bool_array = np.zeros_like(condata, dtype='?')
    big_bool_array[diff_is_large_and_values_far_apart] = True

    edges = np.where(
        np.diff(
            np.append(False,  big_bool_array).astype(int)
        ) == 1
    )[0]

    left_clean = [edges[0]]
    for edge_id in range(len(edges)-1, 0, -1):
        if edges[edge_id] - edges[edge_id-1] > 30:
            left_clean.append(edges[edge_id])

    right_clean = [edges[-1]]
    for edge_id in range(len(edges)-1):
        if edges[edge_id+1] - edges[edge_id] > 80:
            right_clean.append(edges[edge_id])

    surviving_edges = set(right_clean).intersection(left_clean)

    for edge_pos in surviving_edges:

        min1p = edge_pos
        min1 = condata[edge_pos]
        min2p = condata[edge_pos+30:edge_pos+80].argmin()
        min2 = condata[min2p]

        step = (min2 - min1) / (min2p - min1p)

        integral = np.trapz(condata[edge_pos:edge_pos+30])
        # correct for baseline
        integral -= (condata[edge_pos] + (step * 30/2)) * 30

        integrals.append(integral)

    if debug_plotting:
        plt.ion()
        plt.close('all')
        plt.figure()
        plt.plot(cleandata, '.:', label='cleandata')
        plt.plot(condata, '.:', label='condata')
        for e in edges:
            plt.axvline(e, c='k')

        for e in surviving_edges:
            plt.axvline(e, c='r')
        plt.grid()
        plt.legend()
        plt.show()
        input('?')

plt.ion()
plt.hist(
    integrals,
    bins=200,
    range=[0, 1000],
    log=False,
    histtype='step'
)
plt.show()
