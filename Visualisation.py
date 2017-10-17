import os
import errno
import math
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

def gauss(x, coeffs):
    return coeffs[0] * np.exp( -(x - coeffs[1])**2 / (2. * coeffs[2]**2))

def gauss_int(coeffs):
    x = np.linspace(0,1000,1000)
    int_sum = np.trapz(gauss(x, coeffs))
    return int_sum

def gauss_int2(coeffs):
    x = np.linspace(0, 1000, 1000)
    int_sum = np.trapz(coeffs[0] * np.exp( -(x-coeffs[1])**2 / (2. * coeffs[2]**2) ) * (x - coeffs[1])**2 / (coeffs[2]**3))
    return int_sum

def gauss_int3(coeffs):
    x = np.linspace(0, 1000, 1000)
    int_sum = np.trapz(coeffs[0] * np.exp( -(x - coeffs[1])**2 / (2. * coeffs[2]**2)) * (x - coeffs[1]) / (coeffs[2]**2))
    return int_sum

def gauss_sump(x, *coeffs):
    '''Gaussian function used for fitting'''
    return coeffs[0] * np.exp( -(x - coeffs[1])**2 / (2. * coeffs[2]**2)) + coeffs[3] * np.exp( -(x - coeffs[4])**2 / (2. * coeffs[5]**2))


def gauss_sum(x, *coeffs):
    '''Gaussian function used for fitting'''
    return coeffs[0] * np.exp( -(x - coeffs[1])**2 / (2. * coeffs[2]**2)) + coeffs[3] * np.exp( -(x - coeffs[4])**2 / (2. * coeffs[5]**2)) + coeffs[6] * np.exp( -(x - coeffs[7])**2 / (2. * coeffs[8]**2))


def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r+').readlines()
    lines[line_num] = text
    out = open(file_name, 'w+')
    out.writelines(lines)
    out.close()
    
def make_template():
    template = np.append(
    	np.zeros(10),
        sipm_vs_t(2e9, 200, 0)[0:30],
    )
    template /= template.sum()
    return template

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


def analyse(diode, point_num, voltage, temperature, cone, debug_plotting):
    path = f"Data/Diode_{diode}/T_{diode}_{cone}_{point_num}.npy"
    data = np.load(path)

    template = make_template()
    integrals = []
    for event_id, cleandata in tqdm(enumerate(data[:, 20:-20])):
        if event_id > 60:
            break


        condata = np.convolve(template, cleandata, mode='same')




        dcondata = np.convolve(
            np.diff(condata),
            np.ones(10) / 10.,
            'same'
        )

        diff_is_large_and_values_far_apart = []
        for large_derivative_pos in np.where(dcondata[25:-50] > 0.4)[0] + 25: #modded
            i = large_derivative_pos
            _min = condata[i-25:i].min()
            _max = condata[i:i+50].max()
            if abs(_max - _min) < 15: #modded
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


        showfig=plt.figure()
        plt.plot(cleandata, 'k.--')
        plt.plot(condata, 'r.:')
        for edge in surviving_edges:
            plt.axvline(edge,color='r')
        plt.xlabel("Time [0.5 ns]")
        plt.ylabel("Voltage [mV]")
        showfig.savefig(f"Results/cleandata_{event_id}.png")
        plt.close()



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
    fig = plt.figure(1)
    n, bins, patches = plt.hist(
        integrals,
        bins = 200,
        range = [0, 1000],
        log = False,
        histtype = 'step'
    )

    max1p = np.argmax(n)
    min1p = np.argmin(n[max1p:max1p + 30]) + max1p
    max2p = np.argmax(n[min1p:]) + min1p
    min2p = np.argmin(n[max2p:max2p + 30]) + max2p

    max1 = bins[max1p]
    min1 = bins[min1p]
    max2 = bins[max2p]
    min2 = bins[min2p]

    bin_centres = (bins[:-1] + bins[1:]) / 2

    #Fitting first and second peak
    bin_centres = (bins[:-1] + bins[1:]) / 2

    p0t = [n[max1p], max1, 10., n[max2p], max2, 10.]

    coefft, covt = curve_fit(gauss_sump, bin_centres[max1p - 20:min2p], n[max1p - 20:min2p], p0 = p0t)

    gain_est = coefft[4] - coefft[1]
    gain_estp = int(gain_est * 200 / 1000)


    p0 = [n[max1p], max1, 10., n[max2p], max2, 10., n[max2p + gain_estp], max2 + gain_est, 10.]

    coeff, cov = curve_fit(gauss_sum, bin_centres[max1p - 20:], n[max1p - 20:], p0 = p0)

    norm1, mean1, std1, norm2, mean2, std2, norm3, mean3, std3 = coeff

    hist_fit = gauss_sum(bin_centres, *coeff)
    single_fit1 = gauss(bin_centres, [norm1, mean1, std1])
    single_fit2 = gauss(bin_centres, [norm2, mean2, std2])
    single_fit3 = gauss(bin_centres, [norm3, mean3, std3])

    cross_mean1 = gauss_int([norm1, mean1, std1])
    cross_mean2 = gauss_int([norm2, mean2, std2])
    
    cross_mean1_err = np.sqrt(abs(gauss_int2([norm1, mean1, std1]))**2 * cov[2,2] +
    abs(gauss_int3([norm1, mean1, std1]))**2 * cov[1,1] +
    abs(cross_mean1 / norm1)**2 * cov[0,0]
    	)

    cross_mean2_err = np.sqrt(abs(gauss_int2([norm2, mean2, std2]))**2 * cov[5,5] +
    abs(gauss_int3([norm2, mean2, std2]))**2 * cov[4,4] +
    abs(cross_mean2 / norm2)**2 * cov[3,3]
    	)

    cross_mean = cross_mean2 / cross_mean1

    cross_err = cross_mean * np.sqrt((cross_mean1_err / cross_mean1)**2 + (cross_mean2_err / cross_mean2)**2)

    cross_err_alt = cross_mean * (
                            abs(gauss_sum(mean1 + std1, *coeff) - cross_mean1) / cross_mean1 +
                            abs(gauss_sum(mean2 + std2, *coeff) - cross_mean2) / cross_mean2
                            )
    gain = mean2 - mean1
    gain_err = np.sqrt(cov[1,1] + cov[4,4])
    gain_err_alt = std1 / np.sqrt(cross_mean1) + std2 / np.sqrt(cross_mean2)

    plt.axvline(mean1, c = 'g', label = "Gaussian Mean")
    plt.axvline(mean2, c = 'g')
    plt.axvline(bins[max1p], c = 'k', label = "Init Max")
    plt.axvline(bins[max2p], c = 'k')
    plt.axvline(bins[min1p], c = 'y', label = "Init Min")
    plt.axvline(bins[min2p], c = 'y')
    plt.axvline(bins[max2p + gain_estp], c = 'k')

    plt.plot(bin_centres, hist_fit, c = 'r')
    plt.plot(bin_centres, single_fit1, c = 'rosybrown', alpha = 0.3)
    plt.plot(bin_centres, single_fit2, c = 'indianred', alpha = 0.3)
    plt.plot(bin_centres, single_fit3, c = 'brown', alpha = 0.3)
    plt.xlabel("p.e.")
    plt.ylabel("# of events")
    plt.legend()
    plt.title(f"Vol: {voltage} V, Temp: {temperature}, Gain: {mean2-mean1}")
    plt.show()

    print(f"1 Photon events: {cross_mean1}")
    print(f"2 Photon events: {cross_mean2}")
    print(f"Crosstalk Probability: {cross_mean*100} +/- {cross_err*100}%")
    print(f"Gain: {gain} +/- {gain_err}")

#Adjust those !!!
#----------------------------------------
diode = 1
point_num = [1, 2, 3, 4, 5]
voltage_True = [69.0095, 69.3829, 69.7102, 69.9180, 70.1993]
voltage_False = [69.0970, 69.3815, 69.7096, 69.9196, 70.2019]
temperature_True = [5.12, 10.12, 15.13, 20.15, 25.18]
temperature_False = [5.09, 10.09, 15.12, 20.12, 25.18]
debug_plotting = False
#----------------------------------------


analyse(diode, 2, 23, 5, True, debug_plotting)

