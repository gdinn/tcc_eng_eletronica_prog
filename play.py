#medir
# 50k, 100k, 200k ... 1M, 5M, 10M, 15M, 25M

#bom q tem ganho no SDR
# pode ser q o play tenha uma curva diferente e por isso a medida em menor freq esta com menor magnitude

# verificar medida em baixa frequencia

from pylab import *
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import control
from time import sleep
import SoapySDR as sp

medir_freq = 25e6 #min value = 10e3 // max value = 1.7e9
n_ensaios = 1

gain = 0
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

sample_rate = 500e3
gain = 42 - gain
center_freq = medir_freq + sample_rate/10

args = dict(driver="sdrplay")
sdr = sp.Device(args)
rx_chan = 0
sdr.setSampleRate(sp.SOAPY_SDR_RX, rx_chan, sample_rate)
sdr.setFrequency(sp.SOAPY_SDR_RX, rx_chan, center_freq)
sdr.setGainMode(sp.SOAPY_SDR_RX, rx_chan, False) # turn off AGC (automatic gain control)
sdr.setGain(sp.SOAPY_SDR_RX, rx_chan, gain)
# max_nsamples = 32256 # absolute max value
max_nsamples = 128
measurement_time = 1000e-3


rxStream = sdr.setupStream(sp.SOAPY_SDR_RX, sp.SOAPY_SDR_CF32)

if debug:
    print("faxa de ganho")
    print(sdr.getGainRange(sp.SOAPY_SDR_RX, rx_chan))
    print("[DONE]setup")

magnitudes = []
frequencies = []
samples = [[0]]*n_ensaios

sdr.activateStream(rxStream) #start streaming
print("aquisição %ds * %d ensaios" % (measurement_time, n_ensaios)) 
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
print("\n%d ensaios, aquisição=%fs (%d PSDs/medida)" % (n_ensaios, measurement_time, medidas_ensaio))
print("------------------------------")
for i in range(n_ensaios):
    #samples[i] = samples[i] - np.mean(samples[i])
    freq, mag = obter_magnitude(
		freq = medir_freq,
		samples = samples[i],
		sample_rate = sample_rate,
		center_freq = center_freq
	)

    magnitudes.append(mag)
    frequencies.append(freq)
    print("Na frequência", freq, "a magnitude é de", mag, "dB")

fazer_grafico(
    freq = medir_freq,
    samples = samples[0],
    sample_rate = sample_rate,
    center_freq = center_freq      
)

frequencies_str = ', '.join([str(elem) for elem in frequencies])
f = open("frequencies_play.txt", "w")
f.write(frequencies_str)
f.close()

magnitudes_str = ', '.join([str(elem) for elem in magnitudes])
f = open("magnitudes_play.txt", "w")
f.write(magnitudes_str)
f.close()