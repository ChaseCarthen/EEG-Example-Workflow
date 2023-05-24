import mne
import numpy as np
import pandas as pd
import os
from ssqueezepy import ssq_cwt, ssq_stft, cwt, icwt
from ssqueezepy.visuals import plot, imshow
from ssqueezepy.wavelets import Wavelet, center_frequency
from ssqueezepy.experimental import scale_to_freq,freq_to_scale

os.environ['SSQ_PARALLEL'] = '1'


def parseEvents(df):
    events = []
    for index,row in df.iterrows():
        if index % 2 == 0:
            events.append({})
            events[int(index/2)]['start'] = row['timestamp']
            events[int(index/2)]['eventname'] = row['event_id']
        else:
            events[int(index/2)]['end'] = row['timestamp']
    return events

# event should be {'eventname':'somename',start:0,end:some number greater than start}
def parseEventData(data, event, startOfExperiment, endOfExperiment):
    startRatio = (event['start'] - startOfExperiment)/(endOfExperiment - startOfExperiment)
    endRatio = (event['end']- startOfExperiment)/(endOfExperiment - startOfExperiment)
    start = int(startRatio * data.shape[1])
    end = int(endRatio * data.shape[1])
    return data[:,start:end]

def processData(signal,targetData,label=0):
    for i in [channels.index(channel) for channel in channels]:
        bandData = extractBands(signal[i])
        if not str(i) + 'deltapower' in data.keys():
            targetData[str(i) + 'deltapower'] = []
        if not str(i) + 'thetapower' in data.keys():
            targetData[str(i) + 'thetapower'] = []
        if not str(i) + 'gammapower' in data.keys():
            targetData[str(i) + 'gammapower'] = []
        if not str(i) + 'alphapower' in data.keys():
            targetData[str(i) + 'alphapower'] = []
        if not str(i) + 'betapower' in data.keys():
            targetData[str(i) + 'betapower'] = []
        targetData[str(i) +'deltapower'].append(bandData['delta']['power'])
        targetData[str(i) +'thetapower'].append(bandData['theta']['power'])
        targetData[str(i) +'gammapower'].append(bandData['gamma']['power'])
        targetData[str(i) +'alphapower'].append(bandData['alpha']['power'])
        targetData[str(i) +'betapower'].append(bandData['beta']['power'])
        #print('test')
        targetData['labels'] = label
    return targetData

def getScaleForFrequency(wavelet=Wavelet('morlet'),fromFreq=1,toFreq=3, fs=128.0, searchScales=100):
    #print([fromFreq/fs,toFreq/fs])
    output = freq_to_scale(np.array([fromFreq,toFreq]),wavelet,wavelet.N,fs=fs, n_search_scales=searchScales)
    output = output[::-1]
    scale = (output[1] - output[0]) / float(searchScales)
    output = np.arange(output[0],output[1],scale)[::-1]
    return output

def computeMaximumPower(wavelet):
    return np.sum(10 * np.log10(2 * np.abs(wavelet) ** 2 + .000000001))


# fs is sampling frequency
def extractBands(signal, wavelet=Wavelet('morlet'), fs=128.0):
    bandData = {}
    
    deltaScales = getScaleForFrequency(fromFreq=1,toFreq=3,wavelet=wavelet,fs=fs)
    thetaScales = getScaleForFrequency(fromFreq=4,toFreq=7,wavelet=wavelet,fs=fs)
    alphaScales = getScaleForFrequency(fromFreq=8,toFreq=12,wavelet=wavelet,fs=fs)
    betaScales = getScaleForFrequency(fromFreq=13,toFreq=25,wavelet=wavelet,fs=fs)
    gammaScales = getScaleForFrequency(fromFreq=30,toFreq=45,wavelet=wavelet,fs=fs)

    allScales = getScaleForFrequency(fromFreq=1,toFreq=45,searchScales=1000)
    wavelet = Wavelet('morlet')
    allFreqs = scale_to_freq(allScales, wavelet, wavelet.N, fs=fs)
    deltaIndexs = np.where(allFreqs <= 3)
    thetaIndexs = np.where((allFreqs >= 4) & (allFreqs <= 7))
    alphaIndexs = np.where((allFreqs >= 8) & (allFreqs <= 12))
    betaIndexs = np.where((allFreqs >= 13) & (allFreqs <= 25))
    gammaIndexs = np.where((allFreqs >= 30) & (allFreqs <= 45))
     
    # running the wavelet
    coef,scales = cwt(signal,wavelet,allScales,fs=fs)

    # reverses the wavelet
    alphaSignal = icwt(coef[alphaIndexs], scales=allScales[alphaIndexs])
    deltaSignal = icwt(coef[deltaIndexs], scales=allScales[deltaIndexs])
    betaSignal = icwt(coef[betaIndexs], scales=allScales[betaIndexs])
    gammaSignal = icwt(coef[gammaIndexs], scales=allScales[gammaIndexs])
    thetaSignal = icwt(coef[thetaIndexs], scales=allScales[thetaIndexs])
    
    # gets the coefficients 
    deltaWavelets = coef[deltaIndexs]
    thetaWavelets = coef[thetaIndexs]
    alphaWavelets = coef[alphaIndexs]
    betaWavelets = coef[betaIndexs]
    gammaWavelets = coef[gammaIndexs]

    #deltaSignal, deltaWavelets, scales = extractSignal(signal, wavelet, deltaScales, fs=fs)
    bandData['delta'] = {'signal': deltaSignal,'wavelet': deltaWavelets, 'power': computeMaximumPower(deltaWavelets)}

    #thetaSignal, thetaWavelets, scales = extractSignal(signal, wavelet, thetaScales, fs=fs)
    bandData['theta'] = {'signal': thetaSignal,'wavelet': thetaWavelets, 'power': computeMaximumPower(thetaWavelets)}

    #alphaSignal, alphaWavelets, scales = extractSignal(signal, wavelet, alphaScales, fs=fs)
    bandData['alpha'] = {'signal': alphaSignal,'wavelet': alphaWavelets, 'power': computeMaximumPower(alphaWavelets)}

    #betaSignal, betaWavelets, scales = extractSignal(signal, wavelet, betaScales, fs=fs)
    bandData['beta'] = {'signal': betaSignal,'wavelet': betaWavelets, 'power': computeMaximumPower(betaWavelets)}

    #gammaSignal, gammaWavelets, scales = extractSignal(signal, wavelet, gammaScales, fs=fs)
    bandData['gamma'] = {'signal': gammaSignal,'wavelet': gammaWavelets, 'power': computeMaximumPower(gammaWavelets)}
    
    return bandData



# load data
events = pd.read_csv('./Emotional-ColorsEvents.csv')
parsedEvents = parseEvents(events)
data = pd.read_csv('./Emotional-ColorsEmotivDataStream-EEG.csv')


data.columns = data.columns.str.replace("'",'')

startOfExperiment = data['WrittenTimestamp'].min()
endOfExperiment = data['WrittenTimestamp'].max()

# create the channels in mne -TODO make a function -- difficulty eash
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']



# Need to know what is the sampling frequency -- emotiv is 128, openBCI? 
info = mne.create_info(ch_names=channels,ch_types='eeg',sfreq=128)

data = data[channels].transpose()
data = mne.io.RawArray(data,info)
data = data.filter(1,45)
#data.plot(scalings={'eeg':100})

filteredData = data.get_data()

range = filteredData.max() - filteredData.min()
min = filteredData.min()
dataRangeMin = range * .02 + min
dataRangeMax = range * .98 + min


filteredData[filteredData >= dataRangeMax] = 0
filteredData[filteredData <= dataRangeMin] = 0

columnNames = [] #['alphapower','deltapower','betapower','gammapower','thetapower']
data = {}

# run on one channel
for event in parsedEvents:
    print(event['eventname'])
    eventData = parseEventData(filteredData, event, startOfExperiment, endOfExperiment)
    data = processData(eventData,data)

df = pd.DataFrame(data,columns=data.keys())
df.to_csv('test.csv', index=False)