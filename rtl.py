from pylab import *
from rtlsdr import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import control

sample_rate = 1e6
center_freq = 92.1e6 + 250e3
measure_freq = center_freq - 250e3
freq_correction = 60
gain = 1

def fazer_aquisicao(sdr):
    samples = sdr.read_samples(1024*512)
    return samples
    
def obter_magnitude(freq, samples, sample_rate, center_freq, measure_freq):
    if sample_rate < 1e6:
        print("sample_rate precisa ser maior igual a 1e6")
        return
    if center_freq < 1e6:
        print("center_freq precisa ser maior igual a 1e6")
        return
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

def obter_magnitude_sem_ganho(mag, ganho):
    amp = control.db2mag(mag)/ganho
    return 20*np.log10(amp)

sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.freq_correction = freq_correction
sdr.gain = gain

magnitudes = []
frequencies = []

for i in range(10):
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
