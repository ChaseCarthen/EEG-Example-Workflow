from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import os
from matplotlib.colors import ListedColormap
from utils import processSignal
from readDeapData import loadDeapFile
import re

# process data as stft, number of seconds to use for the prcoessing
stft = True
seconds = 125*5

labelsMap = {"C1-I1-Start":0,
             "C1-I2-Start":0,
             "C1-I3-Start":0,
             "C1-I4-Start":0,
             "C1-I5-Start":0,
             "C1-I6-Start":0,
             "C1-I7-Start":0,
             "C1-I8-Start":0,
             "C1-I9-Start":0,
             "C1-I10-Start":0,
             "C1-I11-Start":0,
             "C1-I12-Start":0,
             "C2-I1-Start":1,
             "C2-I2-Start":1,
             "C2-I3-Start":1,
             "C2-I4-Start":1,
             "C2-I5-Start":1,
             "C2-I6-Start":1,
             "C2-I7-Start":1,
             "C2-I8-Start":1,
             "C2-I9-Start":1,
             "C2-I10-Start":1,
             "C2-I11-Start":1,
             "C2-I12-Start":1,
             "C3-I1-Start":2,
             "C3-I2-Start":2,
             "C3-I3-Start":2,
             "C3-I4-Start":2,
             "C3-I5-Start":2,
             "C3-I6-Start":2,
             "C3-I7-Start":2,
             "C3-I8-Start":2,
             "C3-I9-Start":2,
             "C3-I10-Start":2,
             "C3-I11-Start":2,
             "C3-I12-Start":2,
             "C4-I1-Start":3,
             "C4-I2-Start":3,
             "C4-I3-Start":3,
             "C4-I4-Start":3,
             "C4-I5-Start":3,
             "C4-I6-Start":3,
             "C4-I7-Start":3,
             "C4-I8-Start":3,
             "C4-I9-Start":3,
             "C4-I10-Start":3,
             "C4-I11-Start":3,
             "C4-I12-Start":3,
          "1-1-Start":4,
          "1-2-Start":4,
          "1-3-Start":4,
          "1-4-Start":4,
          "1-5-Start":4,
          "2-1-Start":5,
          "2-2-Start":5,
          "2-3-Start":5,
          "2-4-Start":5,
          "2-5-Start":5,
          "3-1-Start":6,
          "3-2-Start":6,
          "3-3-Start":6,
          "3-4-Start":6,
          "3-5-Start":6,
          "4-1-Start":7,
          "4-2-Start":7,
          "4-3-Start":7,
          "4-4-Start":7,
          "4-5-Start":7,
          "5-1-Start":8,
          "5-2-Start":8,
          "5-3-Start":8,
          "5-4-Start":8,
          "5-5-Start":8,
          "6-1-Start":9,
          "6-2-Start":9,
          "6-3-Start":9,
          "6-4-Start":9,
          "6-5-Start":9,
          "7-1-Start":10,
          "7-2-Start":10,
          "7-3-Start":10,
          "7-4-Start":10,
          "7-5-Start":10,
          "8-1-Start":11,
          "8-2-Start":11,
          "8-3-Start":11,
          "8-4-Start":11,
          "8-5-Start":11,
          "9-1-Start":12,
          "9-2-Start":12,
          "9-3-Start":12,
          "9-4-Start":12,
          "9-5-Start":12,
          "10-1-Start":13,
          "10-2-Start":13,
          "10-3-Start":13,
          "10-4-Start":13,
          "10-5-Start":13,
          "11-1-Start":14,
          "11-2-Start":14,
          "11-3-Start":14,
          "11-4-Start":14,
          "11-5-Start":14,
          "12-1-Start":15,
          "12-2-Start":15,
          "12-3-Start":15,
          "12-4-Start":15,
          "12-5-Start":15,
          "13-1-Start":16,
          "13-2-Start":16,
          "13-3-Start":16,
          "13-4-Start":16,
          "13-5-Start":16,
          "14-1-Start":17,
          "14-2-Start":17,
          "14-3-Start":17,
          "14-4-Start":17,
          "14-5-Start":17,
          "15-1-Start":18,
          "15-2-Start":18,
          "15-3-Start":18,
          "15-4-Start":18,
          "15-5-Start":18,
          "16-1-Start":19,
          "16-2-Start":19,
          "16-3-Start":19,
          "16-4-Start":19,
          "16-5-Start":19,
          "1-Start":20, # grey screen start for
          "2-Start":20,
          "3-Start":20,
          "4-Start":20,
          "5-Start":20,
          "6-Start":20,
          "7-Start":20,
          "8-Start":20,
          "9-Start":20,
          "10-Start":20,
          "11-Start":20,
          "12-Start":20,
          "13-Start":20,
          "14-Start":20,
          "15-Start":20,
          "16-Start":20,
          "Grey-Screen-1-Start":20,
          "Grey-Screen-2-Start":20,
          "Grey-Screen-3-Start":20,
          "Grey-Screen-4-Start":20,
          "Eyes-Open-Start":21,
          "Eyes-Closed-Start":22
          }

labelsOfInterest = [
                    "C1",
                    "C2", 
                    "C3",
                    "C4",
                    "1-1-Start",
                    "1-2-Start",
                    "1-3-Start",
                    "1-4-Start",
                    "1-5-Start",
                    "2-1-Start",
                    "2-2-Start",
                    "2-3-Start",
                    "2-4-Start",
                    "2-5-Start",
                    "3-1-Start",
                    "3-2-Start",
                    "3-3-Start",
                    "3-4-Start",
                    "3-5-Start",
                    "4-1-Start",
                    "4-2-Start",
                    "4-3-Start",
                    "4-4-Start",
                    "4-5-Start",
                    "5-1-Start",
                    "5-2-Start",
                    "5-3-Start",
                    "5-4-Start",
                    "5-5-Start",
                    "6-1-Start",
                    "6-2-Start",
                    "6-3-Start",
                    "6-4-Start",
                    "6-5-Start",
                    "7-1-Start",
                    "7-2-Start",
                    "7-3-Start",
                    "7-4-Start",
                    "7-5-Start",
                    "8-1-Start",
                    "8-2-Start",
                    "8-3-Start",
                    "8-4-Start",
                    "8-5-Start",
                    "9-1-Start",
                    "9-2-Start",
                    "9-3-Start",
                    "9-4-Start",
                    "9-5-Start",
                    "10-1-Start",
                    "10-2-Start",
                    "10-3-Start",
                    "10-4-Start",
                    "10-5-Start",
                    "11-1-Start",
                    "11-2-Start",
                    "11-3-Start",
                    "11-4-Start",
                    "11-5-Start",
                    "12-1-Start",
                    "12-2-Start",
                    "12-3-Start",
                    "12-4-Start",
                    "12-5-Start",
                    "13-1-Start",
                    "13-2-Start",
                    "13-3-Start",
                    "13-4-Start",
                    "13-5-Start",
                    "14-1-Start",
                    "14-2-Start",
                    "14-3-Start",
                    "14-4-Start",
                    "14-5-Start",
                    "15-1-Start",
                    "15-2-Start",
                    "15-3-Start",
                    "15-4-Start",
                    "15-5-Start",
                    "16-1-Start",
                    "16-2-Start",
                    "16-3-Start",
                    "16-4-Start",
                    "16-5-Start",
                    "C1-I1-Start",
                    "C1-I2-Start",
                    "C1-I3-Start",
                    "C1-I4-Start",
                    "C1-I5-Start",
                    "C1-I6-Start",
                    "C1-I7-Start",
                    "C1-I8-Start",
                    "C1-I9-Start",
                    "C1-I10-Start",
                    "C1-I11-Start",
                    "C1-I12-Start",
                    "C2-I1-Start",
                    "C2-I2-Start",
                    "C2-I3-Start",
                    "C2-I4-Start",
                    "C2-I5-Start",
                    "C2-I6-Start",
                    "C2-I7-Start",
                    "C2-I8-Start",
                    "C2-I9-Start",
                    "C2-I10-Start",
                    "C2-I11-Start",
                    "C2-I12-Start",
                    "C3-I1-Start",
                    "C3-I2-Start",
                    "C3-I3-Start",
                    "C3-I4-Start",
                    "C3-I5-Start",
                    "C3-I6-Start",
                    "C3-I7-Start",
                    "C3-I8-Start",
                    "C3-I9-Start",
                    "C3-I10-Start",
                    "C3-I11-Start",
                    "C3-I12-Start",
                    "C4-I1-Start",
                    "C4-I2-Start",
                    "C4-I3-Start",
                    "C4-I4-Start",
                    "C4-I5-Start",
                    "C4-I6-Start",
                    "C4-I7-Start",
                    "C4-I8-Start",
                    "C4-I9-Start",
                    "C4-I10-Start",
                    "C4-I11-Start",
                    "C4-I12-Start",
                    "1-Start",
                    "2-Start",
                    "3-Start",
                    "4-Start",
                    "5-Start",
                    "6-Start",
                    "7-Start",
                    "8-Start",
                    "9-Start",
                    "10-Start",
                    "11-Start",
                    "12-Start",
                    "13-Start",
                    "14-Start",
                    "15-Start",
                    "16-Start",
                    "Grey-Screen-1-Start",
                    "Grey-Screen-2-Start",
                    "Grey-Screen-3-Start",
                    "Grey-Screen-4-Start",
                    "Eyes-Open-Start",
                    "Eyes-Closed-Start"]

def loadData(datafile,isOpenBci=False):
    data = pd.read_csv(datafile)
    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    data.columns = data.columns.str.replace("'",'') # fixing namings in emotiv data

    if isOpenBci:
        channels = ['Fp1','Fp2','C3','C4','T5','T6','O1','O2','F7','F8','F3','F4','T3','T4','P3','P4']
        #channels=['Fp1','Fp2','F7','F3','F4','F8']
        data[channels] = data[channels] * (4500000)/24/(2**23-1) # scale to uVolts
    return data,channels

def makeEEGNetwork(inputDims,n_components=2):
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=inputDims),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=n_components),
    ])
    encoder.summary()

    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=n_components),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=inputDims[0])
    ])

    embedder = ParametricUMAP(encoder=encoder, n_components=2, dims=inputDims, decoder=decoder, parametric_reconstruction= True,parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),autoencoder_loss = True,metric='correlation')
    return embedder,encoder


def parseEventData(data, event, startOfExperiment, endOfExperiment):
    startRatio = (event['start'] - startOfExperiment)/(endOfExperiment - startOfExperiment)
    endRatio = (event['end']- startOfExperiment)/(endOfExperiment - startOfExperiment)
    start = int(startRatio * data.shape[1])
    end = int(endRatio * data.shape[1])

    print(event['start'],event['end'])
    print(start,end)
    return data[:,start:end]

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

datafile = './openbci/P30/'


def processData(files=[],labelFiles=[],seconds=125*5,stft=True,isOpenBci=True,fmin=13,fmax=30):
    """
    function: processData
    Description: This function processes data generated by gedaps.
    """
    outData = None
    outClassLabels = None
    outParticipants = []
    outChannels = []
    currentClassLabel = 0
    #seconds = 125 * 5 # 5 seconds

    for index in range(len(files)):
        file = files[index]
        eventsfile = labelFiles[index]
        #eventsfile = os.path.dirname(file)+'/Emotional-ColorsEvents.csv'
        event = []
        parsedEvents = []
        if eventsfile != None:
            events = pd.read_csv(eventsfile)
            parsedEvents = parseEvents(events)
        #print(parsedEvents)
        eventsOfInterest = []
        for event in parsedEvents:
            #print(event['eventname'],'filtering')
            #print(event['eventname'])
            #input()
            foundLabel = [(label == event['eventname']) for label in labelsOfInterest]
            found = False
            for labelitem in foundLabel:
                found = found or labelitem
            if found:
                label = labelsOfInterest[foundLabel.index(True)]
                eventsOfInterest.append((event,labelsMap[label]))
        #print(eventsOfInterest)
        data,channels = loadData(file,isOpenBci=isOpenBci)
        startOfExperiment = data['WrittenTimestamp'].min()
        data = data[channels].to_numpy()
        time = 0
        if not stft:
            data = (data - data.mean()) / data.std()
            data = np.array(np.split(data[:int(data.shape[0]/seconds)*seconds,:],int(data.shape[0]/seconds),axis=0))
            data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
        else: 
            outputData,time,fmaxindex,fminindex = processSignal(data)
            if outData != None:
                outData = np.append(outData,outputData,axis=0)
            else:
                outData = outputData
        #     f,t,data = signal.stft(data.T, 125,nperseg=125*5,noverlap=0)
        #     print(f)
        #     print(t)
        #     minf = 1000000
        #     maxf = -100000
        #     fminindex = 100000
        #     fmaxindex = -1
        #     for i in f:
        #         if i >= fmin:
        #             minf = min(i,minf)
        #             fminindex = min(fminindex,np.where(f == i)[0][0])
        #         if i <= fmax:
        #             maxf = max(i,maxf)
        #             fmaxindex = max(np.where(f == i)[0][0],fmaxindex)
            
        #     print(fminindex,fmaxindex)
        #     data = np.abs(data) # turn complex into magnitude
           
        #     data = data[:,fminindex:fmaxindex+1,:] 
        #     for i in range(fmaxindex-fminindex+1):
        #         data[:,i,:] = (data[:,i,:] - data[:,i,:].mean()) / data[:,i,:].std()
        #     time = t
        #     print(data.shape)
        #     data = data.transpose(0,2,1)
        #     splitted = np.split(data,16)
        #     index = 0
        #     for split in splitted:
        #         index += 1
        #         print(split.shape)
        #         splitAppend = split.reshape(split.shape[1],split.shape[2])
        #         if outData is not None:
        #             outData = np.append(outData,splitAppend,axis=0)
        #         else:
        #             outData = splitAppend
        #     print(outData.shape)
        # print(data.shape)
        
        labels = []
        if stft:
            count = 0
            for split in range(16):
                for i in time:
                    found = False
                    index = 0
                    for eventPairing in eventsOfInterest:
                        index += 1
                        start = eventPairing[0]['start'] 
                        end = eventPairing[0]['end']
                        if i >= start-startOfExperiment and i < end-startOfExperiment:
                            labels.append(eventPairing[1])
                            found = True
                            break
                    if not found:
                        labels.append(-1)
                    outChannels.append(count % 16)
                    outParticipants.append(currentClassLabel)
                count += 1
            if outClassLabels is not None:
                outClassLabels = np.append(outClassLabels,labels)
            else:
                outClassLabels = labels
                
        else:
            if outClassLabels is not None:
                outClassLabels = np.append(outClassLabels,[currentClassLabel] * data.shape[0])
            else:
                outClassLabels = [currentClassLabel]*data.shape[0]
        currentClassLabel += 1
    return outData,np.array(outClassLabels), outChannels, outParticipants,fmaxindex-fminindex+1

def plotData(title, data, labels, cmap='tab20', showLabelText=False, save=False, saveName='plot.png'):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 8))

    sc = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        cmap=cmap,
        s=4,
        alpha=0.5,
        rasterized=True
    )

    if showLabelText:
        buckets = []
        for i, label in enumerate(labels):
            if not label in buckets:
                ax.annotate(label,(data[i,0],data[i,1]),textcoords='offset points',xytext=(0,10), ha="center")
                buckets.append(label)

    colorbar = plt.colorbar(sc)
    ax.set_facecolor('white' if showLabelText else 'black')
    ax.axis('equal')
    ax.set_title(title, fontsize=20)
    if save:
        fig.savefig(saveName)

'''
files = ['./Mash-Upsopenbci_eeg_head.csv',
                              './openbci/P24/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P17/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P18/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P19/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P20/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P21/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P4/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P5/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P6/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P7/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P8/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P9/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P10/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P11/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P12/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P13/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P14/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P15/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P16/Emotional-Colorsopenbci_eeg.csv',
                              './Mash-Upsopenbci_eeg_ear.csv'
                              ]
labelFiles = [ None,'./openbci/P24/Emotional-ColorsEvents.csv','./openbci/P16/Emotional-ColorsEvents.csv',None]


data,train_labels,outChannels,outParticipants,dimension = processData(stft=stft,seconds=seconds,files=files,labelFiles=labelFiles)


embedder,encoder = makeEEGNetwork((dimension,))  # 92
embedder.fit_transform(data)
print(embedder._history)

fig, ax = plt.subplots()
ax.plot(embedder._history['loss'])
ax.set_ylabel('Cross Entropy')
ax.set_xlabel('Epoch')


indices = np.where((train_labels < 4) | (train_labels >= 0))
print(indices)
print(train_labels[indices])
print(train_labels.max())
passindata = data
z = encoder.predict(passindata)
#z = z[indices]
'''


colors = ['white', 'red']
colors = ['#FFFFFF', '#FF8000', '#FFFF00', '#80FF00', '#00FF00', '#00FF80', '#00FFFF', '#0080FF',
          '#0000FF', '#8000FF', '#FF00FF', '#FF0080', '#FF3333', '#FF9933', '#FFFF66', '#99FF33']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f4e79', '#ff903d',
          '#358239', '#b9314f', '#903d3d', '#d9b930', '#306a7c', '#64a75a',
          '#b24630', '#303b70', '#a6a6a6', '#ff00ff']
cmap = ListedColormap(colors)

#cmap = 'tab20'

#plotData('UMAP embeddings participants',z,outParticipants)
#plotData('UMAP embeddings channels',z,outChannels)
#plotData('UMAP embeddings labels',z,train_labels,showLabelText=True)


#plt.show()


labelFiles = glob.glob('./openbci/P*/Emotional-ColorsEvents.csv')
files = glob.glob('./openbci/P*/Emotional-Colorsopenbci_eeg.csv')
partcipants = glob.glob('./openbci/P*')


files = glob.glob('/mnt/data3/schooldata/bsonfiles/*.bson')

for file in files:
    base = os.path.basename(file)
    participant = int(re.findall('[0-9]+',base)[0])
    #print(os.path.basename(file))
    outData,outParticipants,outChannel,outLabel = loadDeapFile(file,partcipantNumber=participant)
    print(outData.shape)
    #input()
    dimension = outData.shape[1]
    embedder,encoder = makeEEGNetwork((dimension,))  # 92
    embedder.fit_transform(outData)
    z = embedder.predict(outData)
    plotData('UMAP embeddings participants',z,outParticipants,save=True,saveName='deeppartcipants' + str(participant) + '.png',cmap=cmap)
    plotData('UMAP embeddings channels',z,outChannel,save=True,saveName='deepchannel' + str(participant) + '.png',cmap=cmap)
    plotData('UMAP embeddings labels',z,outLabel,showLabelText=True,save=True,saveName='deeplabel'+ str(participant) + '.png',cmap=cmap)
    print(outData.shape)
    z2 = np.append(z,np.array(outLabel).reshape((len(outLabel),1)),axis=1)
    pd.DataFrame(z2).to_csv('test2.csv',index=False)
    print(z2)
    input()
    
input()
for i in range(len(partcipants)):
    partcipant = partcipants[i]
    file = [partcipants[i]+'/Emotional-Colorsopenbci_eeg.csv']
    labelfile = [partcipants[i]+'/Emotional-ColorsEvents.csv']
    print(partcipant)
    #input()
    data,train_labels,outChannels,outParticipants,dimension = processData(stft=stft,seconds=seconds,files=file,labelFiles=labelfile)

    embedder,encoder = makeEEGNetwork((dimension,))  # 92
    embedder.fit_transform(data)
    print(embedder._history)

    indices = np.where((train_labels < 4) | (train_labels >= 0))
    print(indices)
    print(train_labels[indices])
    print(train_labels.max())
    passindata = data
    z = encoder.predict(passindata)

    #fig, ax = plt.subplots()
    #ax.plot(embedder._history['loss'])
    #ax.set_ylabel('Cross Entropy')
    #ax.set_xlabel('Epoch')

    plotData('UMAP embeddings participants',z,outParticipants,save=True,saveName='partcipants' + os.path.basename(partcipant) + '.png')
    plotData('UMAP embeddings channels',z,outChannels,save=True,saveName='channel' + os.path.basename(partcipant) + '.png')
    plotData('UMAP embeddings labels',z,train_labels,showLabelText=True,save=True,saveName='label'+ os.path.basename(partcipant) + '.png')
