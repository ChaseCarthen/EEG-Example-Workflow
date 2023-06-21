from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import os
from matplotlib.colors import ListedColormap

def loadData(datafile,isOpenBci=False):
    data = pd.read_csv(datafile)
    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    data.columns = data.columns.str.replace("'",'') # fixing namings in emotiv data

    if isOpenBci:
        channels = ['Fp1','Fp2','C3','C4','T5','T6','O1','O2','F7','F8','F3','F4','T3','T4','P3','P4']
        #channels=['Fp1','Fp2']
        data[channels] = data[channels] * (4500000)/24/(2**23-1) # scale to uVolts

    return data,channels

def makeEEGNetwork(inputDims,n_components=2):
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=inputDims),
        #tf.keras.layers.Conv2D(
        #    filters=32, kernel_size=1, strides=(2, 2), activation="relu", padding="same"
        #),
        #tf.keras.layers.Conv2D(
        #    filters=64, kernel_size=1, strides=(2, 2), activation="relu", padding="same"
        #),
        #tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=n_components),
    ])
    #encoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #print(encoder.predict(np.random.random((1,125))))
    encoder.summary()
    #encoder.fit()

    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=n_components),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=inputDims[0])
        #tf.keras.layers.Reshape(target_shape=(7, 7, 256)),
        #tf.keras.layers.UpSampling2D((2)),
        #tf.keras.layers.Conv2D(
        #    filters=64, kernel_size=3, padding="same", activation="relu"
        #),
        #tf.keras.layers.UpSampling2D((2)),
        #tf.keras.layers.Conv2D(
        #    filters=32, kernel_size=3, padding="same", activation="relu"
        #),

    ])

    embedder = ParametricUMAP(encoder=encoder, n_components=2, dims=inputDims, decoder=decoder, parametric_reconstruction= True,parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),autoencoder_loss = True,metric='correlation')
    return embedder,encoder




stft = True
seconds = 125*5
#embedder,encoder = makeEEGNetwork((seconds*16 if not stft else 16*20,)) # 16*129
embedder,encoder = makeEEGNetwork((1*92,))  # 92
#embedder,encoder = makeEEGNetwork((seconds*16,))



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
          "16-5-Start":19
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
                    "C4-I12-Start"]



def makeData(files=[],seconds=125*5,stft=True,isOpenBci=True):
    outData = None
    outClassLabels = None
    outParticipants = []
    outChannels = []
    currentClassLabel = 0
    #seconds = 125 * 5 # 5 seconds

    for file in files:
        eventsfile = os.path.dirname(file)+'/Emotional-ColorsEvents.csv'
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
            f,t,data = signal.stft(data.T, 125,nperseg=125*5,noverlap=0)
            print(f)
            print(t)

            data = np.abs(data) # turn complex into magnitude
           
            data = data[:,61:153,:] 
            for i in range(92):
                data[:,i,:] = (data[:,i,:] - data[:,i,:].mean()) / data[:,i,:].std()
            time = t
            print(data.shape)
            #input()
            #input()
            #print(data.shape)
            data = data.transpose(0,2,1)
            #data = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
            
            splitted = np.split(data,16)
            index = 0
            #del splitted[12:]
            for split in splitted:
                index += 1
                print(split.shape)
                splitAppend = split.reshape(split.shape[1],split.shape[2])
                if outData is not None:
                    outData = np.append(outData,splitAppend,axis=0)
                else:
                    outData = splitAppend
               #input()
            print(outData.shape)

            #data = data.reshape(data.shape[0]*data.shape[1],data.shape[2]).T
        
        print(data.shape)
        
        labels = []
        if stft:
            count = 0
            for split in splitted:
                for i in time:
                    found = False
                    index = 0
                    for eventPairing in eventsOfInterest:
                        index += 1
                        start = eventPairing[0]['start'] 
                        end = eventPairing[0]['end']
                        if i >= start-startOfExperiment and i < end-startOfExperiment:
                            #labels.append(eventPairing[1])
                            #labels.append(index)
                            #print(eventPairing[1])
                            #labels.append(count % 16)
                            labels.append(currentClassLabel)
                            found = True
                    if not found:
                        #labels.append(-1)
                        #labels.append(count % 16)
                        labels.append(currentClassLabel)
                    outChannels.append(count % 16)
                    outParticipants.append(currentClassLabel)
                count += 1
            if outClassLabels is not None:
                outClassLabels = np.append(outClassLabels,labels)
            else:
                outClassLabels = labels
                
        else:
            if outClassLabels is not None:
                outClassLabels = np.append(outClassLabels,[currentClassLabel]*data.shape[0])
            else:
                outClassLabels = [currentClassLabel]*data.shape[0]
        #if outData is not None:
        #    outData = np.append(outData,data,axis=0)
        #else:
        #    outData = data
        print(outData.shape)
        print(len(outClassLabels))
        #input()
        currentClassLabel += 1
    #data,channels = loadData('./Mash-Upsopenbci_eeg_head.csv',isOpenBci=True)
    #data2,channels = loadData('./Mash-Upsopenbci_eeg_ear.csv',isOpenBci=True)
    #data,channels = loadData('./openbci/P24/Emotional-Colorsopenbci_eeg.csv',isOpenBci=True)
    #data2,channels = loadData('./openbci/P27/Emotional-Colorsopenbci_eeg.csv',isOpenBci=True)
    #data3,channels = loadData('./openbci/P26/Emotional-Colorsopenbci_eeg.csv',isOpenBci=True)
    #print(data)

    #data = data[channels].to_numpy()
    #data2 = data2[channels].to_numpy()
    #data3 = data3[channels].to_numpy()

    #print(data.shape)

    #print(data[:int(data.shape[0]/125)*125,:].shape[0]/125)


    #print(type(data))
    #print(data.shape)



    #data = np.array(np.split(data[:int(data.shape[0]/125)*125,:],int(data.shape[0]/125),axis=0))
    #data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    #data2 = np.array(np.split(data2[:int(data2.shape[0]/125)*125,:],int(data2.shape[0]/125),axis=0))
    #data2 = data2.reshape(data2.shape[0],data2.shape[1]*data2.shape[2])
    #data3 = np.array(np.split(data3[:int(data3.shape[0]/125)*125,:],int(data3.shape[0]/125),axis=0))
    #data3 = data3.reshape(data3.shape[0],data3.shape[1]*data3.shape[2])

    #train_labels = np.append(np.zeros(data.shape[0]),np.ones(data2.shape[0]))
    #train_labels = np.append(train_labels,[2]*data3.shape[0])
    #data = np.append(data,data2,axis=0)
    #data = np.append(data,data3,axis=0)

    return outData,np.array(outClassLabels), outChannels, outParticipants


data,train_labels,outChannels,outParticipants = makeData(stft=stft,seconds=seconds,files=['./Mash-Upsopenbci_eeg_head.csv',
                              #'./openbci/P24/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P17/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P18/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P19/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P20/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P21/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P4/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P5/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P6/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P7/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P8/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P9/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P10/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P11/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P12/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P13/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P14/Emotional-Colorsopenbci_eeg.csv',
                              #'./openbci/P15/Emotional-Colorsopenbci_eeg.csv',
                              './openbci/P16/Emotional-Colorsopenbci_eeg.csv',
                              './Mash-Upsopenbci_eeg_ear.csv'
                              ])


embedder.fit_transform(data)


print(embedder._history)
fig, ax = plt.subplots()
ax.plot(embedder._history['loss'])
ax.set_ylabel('Cross Entropy')
ax.set_xlabel('Epoch')


indices = np.where((train_labels < 4) | (train_labels >= 0))
#print(train_labels)
print(indices)
print(train_labels[indices])
print(train_labels.max())
#passindata = data.reshape(data.shape[0],16, 313,1)
passindata = data
z = encoder.predict(passindata)
#z = z[indices]

fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
colors = ['white', 'red']
colors = ['#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#00FF00', '#00FF80', '#00FFFF', '#0080FF',
          '#0000FF', '#8000FF', '#FF00FF', '#FF0080', '#FF3333', '#FF9933', '#FFFF66', '#99FF33']
cmap = ListedColormap(colors)

cmap = 'tab20'
sc = ax.scatter(
    z[:, 0],
    z[:, 1],
    c=outChannels,
    cmap=cmap,
    s=4,
    alpha=0.5,
    rasterized=True
)
colorbar = plt.colorbar(sc)
ax.set_facecolor('black')

ax.axis('equal')
ax.set_title("UMAP embeddings channels", fontsize=20)

fig, ax = plt.subplots(ncols=1, figsize=(10, 8))

sc = ax.scatter(
    z[:, 0],
    z[:, 1],
    c=outParticipants,
    cmap=cmap,
    s=4,
    alpha=0.5,
    rasterized=True
)
colorbar = plt.colorbar(sc)
ax.set_facecolor('black')

ax.axis('equal')
ax.set_title("UMAP embeddings parts", fontsize=20)
#legend_handles = []
#legend_labels = ['Legend Label 1', 'Legend Label 2']
#for label in legend_labels:
#    legend_handles.append(plt.Line2D([], [], color='white', marker='o', markersize=10, label=label, markerfacecolor='black'))
#legend = ax.legend(handles=legend_handles, loc='upper right')





plt.show()