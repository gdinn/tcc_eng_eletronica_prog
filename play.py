from pylab import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import control
from time import sleep
import math
import SoapySDR as sp

sample_rate = 1e6
center_freq = 92.1e6
nsamples = 1024
gain = 1

def fazer_aquisicao(sdr, rxStream):
    sdr.activateStream(rxStream) #start streaming
    print("[DONE]activateStream")    
    _samples = np.array([0]*max_nsamples, np.complex64)
    samples = np.array([], np.complex64)
    mperdidas = 0;
    for j in range(rodadas):        
        for i in range(20):
            sr = sdr.readStream(rxStream, [_samples], len(_samples))
            if sr.ret > 0:                
                break
            if i == 19:
                mperdidas = mperdidas + 1
                print("[ERROR]readStream", sr.ret)
                print("[INFO]", mperdidas, "/", rodadas, "medidas perdidas")
                break
        samples = np.concatenate([samples, _samples])
    sdr.deactivateStream(rxStream) #stop streaming
    return samples

def obter_magnitude(freq, samples, sample_rate, center_freq):
    if sample_rate < 1e6:
        print("sample_rate precisa ser maior igual a 1e6")
        return
    if center_freq < 1e6:
        print("center_freq precisa ser maior igual a 1e6")
        return
    sample_rate = sample_rate/1e6
    center_freq = center_freq/1e6
    freq = freq/1e6
    NFFT = 1024
    axes = psd(samples, NFFT=NFFT, Fs=sample_rate)
    y = 10*np.log10(axes[0])
    x = axes[1] + center_freq
    i = find_nearest(x, freq)    
    return x[i],y[i]

def find_nearest(array, value):
    #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def obter_magnitude_sem_ganho(mag, ganho):
    amp = control.db2mag(mag)/ganho
    return 20*np.log10(amp)



args = dict(driver="sdrplay")
sdr = sp.Device(args)
rx_chan = 0
gain = 42 - gain
sdr.setSampleRate(sp.SOAPY_SDR_RX, rx_chan, sample_rate)
sdr.setFrequency(sp.SOAPY_SDR_RX, rx_chan, center_freq)
sdr.setGainMode(sp.SOAPY_SDR_RX, rx_chan, False) # turn off AGC
sdr.setGain(sp.SOAPY_SDR_RX, rx_chan, gain)
max_nsamples = 32256
rodadas = 1
if nsamples > max_nsamples:
    rodadas = math.ceil(nsamples/max_nsamples)
print("[INFO]Para", nsamples, "serão feitas", rodadas, "medidas")
rxStream = sdr.setupStream(sp.SOAPY_SDR_RX, sp.SOAPY_SDR_CF32)
print("faxa de ganho")
print(sdr.getGainRange(sp.SOAPY_SDR_RX, rx_chan))
print("[DONE]setup")


magnitudes = []
frequencies = []

for i in range(10):
  samples = fazer_aquisicao(
    sdr=sdr,
    rxStream=rxStream
  )
  freq, mag = obter_magnitude(
      freq = center_freq,
      samples = samples,
      sample_rate = sample_rate,
      center_freq = center_freq
  )
  magnitudes.append(mag)
  frequencies.append(freq)
  print("Na frequência", freq, "a magnitude é de", mag, "dB")
sdr.closeStream(rxStream)
sleep(1)
print("[DONE]fechou")

frequencies_str = ', '.join([str(elem) for elem in frequencies])
f = open("frequencies_play.txt", "w")
f.write(frequencies_str)
f.close()

magnitudes_str = ', '.join([str(elem) for elem in magnitudes])
f = open("magnitudes_play.txt", "w")
f.write(magnitudes_str)
f.close()