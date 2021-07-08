from pylab import *
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import control
from time import sleep
import math
import SoapySDR as sp

sample_rate = 1e6
# center_freq = 92.1e6
center_freq = 92.9e6
gain = 1
debug = 0

def fazer_aquisicao(sdr, rxStream):
    _samples = np.array([0]*max_nsamples, np.complex64)
    samples = np.array([], np.complex64)
    end = 0
    start = time.process_time()
    while(end-start < measurement_time):     
        sr = sdr.readStream(rxStream, [_samples], len(_samples))
        if sr.ret > 0:                
            samples = np.concatenate([samples, _samples])
        end = time.process_time()
    #print("\nTempo de aquisição: %.3fms" % ((end-start)*1e3))
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
# max_nsamples = 32256
max_nsamples = 128
measurement_time = 1000e-3
n_ensaios = 10

rxStream = sdr.setupStream(sp.SOAPY_SDR_RX, sp.SOAPY_SDR_CF32)

if debug:
    print("faxa de ganho")
    print(sdr.getGainRange(sp.SOAPY_SDR_RX, rx_chan))
    print("[DONE]setup")

magnitudes = []
frequencies = []
samples = [[0]]*n_ensaios

sdr.activateStream(rxStream) #start streaming
print("aquisição meas_time %ds %d rodadas" % (measurement_time, n_ensaios)) 
for i in range(n_ensaios):
  samples[i] = fazer_aquisicao(
    sdr=sdr,
    rxStream=rxStream
  )
print("aquisição terminou")
sdr.deactivateStream(rxStream) #stop streaming
sleep(1)
sdr.closeStream(rxStream)

medidas_ensaio = 1
print("\n%d ensaios, meas_time=%fs (%d medidas/ensaio)" % (n_ensaios, measurement_time, medidas_ensaio))
print("------------------------------")
for i in range(n_ensaios):
  freq, mag = obter_magnitude(
      freq = center_freq,
      samples = samples[i],
      sample_rate = sample_rate,
      center_freq = center_freq
  )
  magnitudes.append(mag)
  frequencies.append(freq)
  print("Na frequência", freq, "a magnitude é de", mag, "dB")

medidas_ensaio = 10
print("------------------------------\n")
print("%d ensaios, meas_time=%fs (%d medidas/ensaio)" % (n_ensaios, measurement_time, medidas_ensaio))
print("------------------------------")
for i in range(n_ensaios):
    mag2 = []
    freq2 = []    
    tamanho_pedaço = math.floor(len(samples[i])/medidas_ensaio)
    for j in range(medidas_ensaio):
        inicio_samples = j*tamanho_pedaço
        fim_samples = (j+1)*tamanho_pedaço - 1
        _samples = samples[i][inicio_samples:fim_samples]
        freq, mag = obter_magnitude(
            freq = center_freq,
            samples = _samples,
            sample_rate = sample_rate,
            center_freq = center_freq
        )
        mag2.append(mag)
        freq2.append(freq)        
    mag = sum(mag2) / len(mag2)
    magnitudes.append(mag)
    print("Na frequência", freq, "a magnitude é de", mag, "dB")

# frequencies_str = ', '.join([str(elem) for elem in frequencies])
# f = open("frequencies_play.txt", "w")
# f.write(frequencies_str)
# f.close()

# magnitudes_str = ', '.join([str(elem) for elem in magnitudes])
# f = open("magnitudes_play.txt", "w")
# f.write(magnitudes_str)
# f.close()