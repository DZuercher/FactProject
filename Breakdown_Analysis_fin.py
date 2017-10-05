
# coding: utf-8

# In[96]:

import os
import errno
import math
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

def gauss_sum(x, *p):
    '''Gaussian function used for fitting'''
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))+A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

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


# In[102]:

def analyse(diode, point_num, voltage, temperature, debug_plotting):
    path = f"Data/Diode_{diode}/BV_{diode}_{point_num}.npy"
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
    fig=plt.figure(1)
    n, bins, patches = plt.hist(
        integrals,
        bins=200,
        range=[0, 1000],
        log=False,
        histtype='step'
    )

    max1p=np.argmax(n)
    min1p=np.argmin(n[max1p:max1p+30])+max1p
    max2p=np.argmax(n[min1p:])+min1p
    min2p=np.argmin(n[max2p:max2p+30])+max2p

    max1=bins[max1p]
    min1=bins[min1p]
    max2=bins[max2p]
    min2=bins[min2p]

    bin_centres = (bins[:-1] + bins[1:])/2
    fitdata = n[:min2p]

    p0 = [n[max1p], max1, 10., n[max2p], max2, 10.]

    coeff = curve_fit(gauss_sum, bin_centres[:min2p], fitdata, p0=p0)[0]

    hist_fit = gauss_sum(bin_centres[:min2p], *coeff)

    norm1,mean1,std1,norm2,mean2,std2=coeff
    cross_mean2=gauss_sum(mean2, *coeff)
    cross_mean1=gauss_sum(mean1, *coeff)
    cross_mean=cross_mean2 / cross_mean1
    cross_err=cross_mean * (
        abs(gauss_sum(mean1 + std1, *coeff) - cross_mean1) / cross_mean1 + 
        abs(gauss_sum(mean2 + std2, *coeff) - cross_mean2) / cross_mean2
    )


    plt.axvline(mean1,c='g',label="Gaussian Mean")
    plt.axvline(mean2,c='g')
    plt.axvline(bins[max1p],c='k',label="Init Max")
    plt.axvline(bins[max2p],c='k')
    plt.axvline(bins[min1p],c='y',label="Init Min")
    plt.axvline(bins[min2p],c='y')
    plt.plot(bin_centres[:min2p],hist_fit[:min2p],c='r')
    plt.xlabel("p.e.")
    plt.ylabel("# of events")
    plt.legend()
    plt.title(f"Vol: {voltage} V, Temp: {temperature}, Gain: {mean2-mean1}")
    plt.show()


    print("1 Photon events: "+str(cross_mean1))
    print("2 Photon events: "+str(cross_mean2))
    print("Crosstalk Probability: "+str(cross_mean*100)+" +/-"+str(cross_err*100)+" % ")
    print("Gain: "+str(mean2-mean1))


    if os.path.isfile(f"Results/Diode_{diode}/BV/plotdata.txt")==False:
        file=open(f"Results/Diode_{diode}/BV/plotdata.txt","w+")
        file.write("Voltage, Temperature, Gain \n")
        for i in range(100):
            file.write(". \n")
        file.close()
    

    try:
        os.mkdir(f"Results/Diode_{diode}/BV/Point_{point_num}")
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
        
    replace_line(f"Results/Diode_{diode}/BV/plotdata.txt",point_num,f"{voltage} {temperature} {mean2-mean1}")

    fig.savefig(f"Results/Diode_{diode}/BV/Point_{point_num}/Fingerplot_BV_{diode}_{point_num}.png")


# In[104]:

#Adjust those !!!
#Important: Delete plotdata.txt before running
#----------------------------------------
diode = 1
point_num = [1,2,3,4]
voltage = [69.3,69.5,69.7,69.9]
temperature = 25
debug_plotting = False
#----------------------------------------

for i in range(4):
    analyse(diode, point_num[i], voltage[i], temperature, debug_plotting)


# In[105]:

volt = []
gain = []
file = open(f"Results/Diode_{diode}/BV/plotdata.txt", 'r')
for id, line in enumerate(file):
    if id != 0:
        for i, word in enumerate(line.split()):
            if word == ".":
                continue
            if i == 0:
                volt.append(float(word))
            if i == 1:
                temp = word
            if i == 2:
                gain.append(float(word[:-1]))
file.close()

def fitfunc(x,a,b):
    return a+b*x

import scipy

opt, cov = scipy.optimize.curve_fit(fitfunc, volt, gain)
xp = np.linspace(69, 70, 100)
yp = opt[0] + opt[1] * xp
bv = -opt[0] / opt[1]

err_a = np.sqrt(cov[0,0]) 
err_b = np.sqrt(cov[1,1]) 
err_bv = abs(bv * (err_a / opt[0] + err_b / opt[1]))

fig2=plt.figure(2)
plt.plot(xp,yp,'--')
plt.plot(volt,gain,'.')
plt.title(f"Breakdown Voltage Diode {diode}: {bv:.3f} +/- {err_bv:.3f} V")
plt.xlabel("Bias Voltage [V]")
plt.ylabel("Gain")
fig2.show()

fig2.savefig(f"Results/Diode_{diode}/BV/Breakdown_{diode}.png")

print(f"Breakdown Voltage of Diode {diode} is: {bv:.3f} +/- {err_bv:.3f} V")


# In[ ]:



