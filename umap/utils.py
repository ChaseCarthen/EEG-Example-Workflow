from scipy import signal
import numpy as np
def processSignal(data, seconds=125*5,fmin=13,fmax=30,samplerate=125,numChannels=16):
    '''
    name: processSignal
    Description: This function will take in a signal of form: signal samples by channel and will convert it to a short fast fourier transform 
    '''
    if len(data.shape) == 1:
        data = data.reshape(data.shape[0],1)
    outData = None
    f,time,data = signal.stft(data.T, samplerate,nperseg=seconds,noverlap=0)
    minf = 1000000
    maxf = -100000
    fminindex = 100000
    fmaxindex = -1
    for i in f:
        if i >= fmin:
            minf = min(i,minf)
            fminindex = min(fminindex,np.where(f == i)[0][0])
        if i <= fmax:
            maxf = max(i,maxf)
            fmaxindex = max(np.where(f == i)[0][0],fmaxindex)
    data = np.abs(data) # turn complex into magnitude
    data = data[:,fminindex:fmaxindex+1,:] 
    for i in range(fmaxindex-fminindex+1):
        data[:,i,:] = (data[:,i,:] - data[:,i,:].mean()) / data[:,i,:].std()
    #for i in range(0,data.shape[2]):
    #    data[:,:,i] = (data[:,:,i] - data[:,:,i].mean()) / data[:,:,i].std()
    data = data.transpose(0,2,1)
    splitted = np.split(data,numChannels)
    index = 0
    for split in splitted:
        index += 1
        splitAppend = split.reshape(split.shape[1],split.shape[2])
        if outData is not None:
            outData = np.append(outData,splitAppend,axis=0)
        else:
            outData = splitAppend
    return outData,time,fmaxindex,fminindex