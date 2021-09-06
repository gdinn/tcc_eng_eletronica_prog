from pylab import *
from rtlsdr import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import control

sample_rate = 1e6
measure_freq = 92.1e6 # min value = 500e3 // max value = 1,7e9
center_freq = measure_freq + 100e3
freq_correction = 60
gain = 10

def fazer_aquisicao(sdr):
    samples = sdr.read_samples(1024*512)
    return samples
    
def obter_magnitude(freq, samples, sample_rate, center_freq, measure_freq):
    sample_rate = sample_rate/1e6
    center_freq = center_freq/1e6
    measure_freq = measure_freq/1e6
    freq = freq/1e6
    NFFT = 1024
    axes = psd(samples, NFFT=NFFT, Fs=sample_rate)
    y = 10*np.log10(axes[0])
    x = axes[1] + center_freq
    i = find_nearest(x, measure_freq)    
    return x[i],y[i]

def find_nearest(array, value):
    #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def fazer_grafico(freq, samples, sample_rate, center_freq):
    NFFT = 1024
    plt.figure() 
    
    axes = plt.psd(samples, NFFT=NFFT, Fs=sample_rate/1e6, Fc=center_freq/1e6, zorder=10, color='crimson')
    x = np.ones(len(axes[0]))*freq/1e6
    dx = (((sample_rate)/1e6)/2)/25
    plt.plot(x+dx, 10*np.log10(axes[0]), color='pink', linestyle='dashed', zorder=5)
    plt.plot(x-dx, 10*np.log10(axes[0]), color='pink', linestyle='dashed', zorder=5)


    xlabel('Frequency (MHz)')
    ylabel('Relative power (dB)') 
    
    plt.savefig("psd.png")


sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.freq_correction = freq_correction
# sdr.set_direct_sampling('q')
sdr.gain = gain

magnitudes = []
frequencies = []

for i in range(1):
  samples = fazer_aquisicao(
    sdr=sdr
  )
  samples = samples - np.mean(samples)
  freq, mag = obter_magnitude(
      freq = center_freq,
      samples = samples,
      sample_rate = sample_rate,
      center_freq = center_freq,
      measure_freq = measure_freq
  )
  fazer_grafico(
    freq = measure_freq,
    samples = samples,
    sample_rate = sample_rate,
    center_freq = center_freq      
  )
  magnitudes.append(mag)
  frequencies.append(freq)  
  print("Na frequência", freq, "a magnitude é de", mag, "dB")
sdr.close()

frequencies_str = ', '.join([str(elem) for elem in frequencies])
f = open("frequencies_rtl.txt", "w")
f.write(frequencies_str)
f.close()

magnitudes_str = ', '.join([str(elem) for elem in magnitudes])
f = open("magnitudes_rtl.txt", "w")
f.write(magnitudes_str)
f.close()
