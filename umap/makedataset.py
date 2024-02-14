from umap.parametric_umap import ParametricUMAP
from umap import UMAP
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
from readOpenBCI import processData
import yaml
def makeEEGNetwork(inputDims,n_components=2,og=True):
    if og:
        embedder = UMAP(n_components=n_components,metric='euclidean')
        return embedder, None
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=inputDims),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=n_components, activation='relu'),
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


def makeDataSet(path='./dataparticipantsdeap/data/*.bson',labels='./*',is_deap=True,allChannel=False,oneParticipant=True,seconds=5,labelsMap={}):
    files = glob.glob(path)
    #files = ['/mnt/data3/schooldata/data/s04.bson','/mnt/data3/schooldata/data/s11.bson','/mnt/data3/schooldata/data/s23.bson']

    allOutData=None
    allOutParticipants=None 
    allOutChannel=None
    allOutArousal=None
    allOutValence=None
    allOutDominance=None
    allOutTime=None
    allOutTrial=None
    embedder = None
    encoder = None
    count = 0
    #allChannel = False
    #oneParticipant = True
    if is_deap:
        for file in files:
            
            base = os.path.basename(file)
            participant = int(re.findall('[0-9]+',base)[0])
            #print(os.path.basename(file))
            
            outData, outParticipants,outChannel,outArousal,outValence,outDominance,outTrial,outTime = loadDeapFile(file,partcipantNumber=participant,allChannel=allChannel,fmin=0,fmax=100,useGroupsRating=False,seconds=seconds)

            if allOutData is not None:
                allOutData = np.append(allOutData,outData,axis=0)
                allOutParticipants = np.append(allOutParticipants,outParticipants,axis=0)
                allOutChannel = np.append(allOutChannel,outChannel,axis=0)
                allOutDominance = np.append(allOutDominance,outDominance,axis=0)
                allOutArousal = np.append(allOutArousal,outArousal,axis=0)
                allOutValence = np.append(allOutValence,outValence,axis=0)
                allOutTrial = np.append(allOutTrial,outTrial,axis=0)
                allOutTime = np.append(allOutTime,outTime,axis=0)
            else:
                allOutData = outData
                allOutParticipants = outParticipants
                allOutChannel = outChannel
                allOutDominance = outDominance
                allOutArousal = outArousal
                allOutValence = outValence
                allOutTrial = outTrial
                allOutTime = outTime

            print(outData.shape)

            print(outArousal.shape)
            #input()
            #input()
            dimension = outData.shape[1]
            #if embedder is None and encoder is None:
            if oneParticipant:
                embedder,encoder = makeEEGNetwork((dimension,),og=True)  # 92
                embedder.fit_transform(allOutData)
            

                z = embedder.transform(allOutData)
                z2 = np.append(z,np.array(allOutArousal).reshape((len(allOutArousal),1)),axis=1)
                z2 = np.append(z2,np.array(allOutValence).reshape((len(allOutValence),1)),axis=1)
                z2 = np.append(z2,np.array(allOutDominance).reshape((len(allOutDominance),1)),axis=1)
                z2 = np.append(z2,np.array(allOutTrial).reshape((len(allOutTrial),1)),axis=1)
                z2 = np.append(z2,np.array(allOutParticipants).reshape((len(allOutParticipants),1)),axis=1)
                z2 = np.append(z2,np.array(allOutChannel).reshape((len(allOutChannel),1)),axis=1)
                z2 = np.append(z2,np.array(allOutTime).reshape((len(allOutTime),1)),axis=1)
                print(z2)
                filename = './' + ('alldata' if not allChannel else 'tryallchannelonepart') + '/'+ base + '.csv'
                print(filename)
                pd.DataFrame(z2).to_csv(filename,index=False,header=['data_1','data_2','arousal','valence','dominance','trial','participant','channel','time'])

                allOutData=None
                allOutParticipants=None 
                allOutChannel=None
                allOutArousal=None
                allOutValence=None
                allOutDominance=None
                allOutTime=None
                allOutTrial=None
            count += 1

        if not oneParticipant:
            embedder,encoder = makeEEGNetwork((dimension,),og=True)  # 92
            embedder.fit_transform(allOutData)


            z = embedder.transform(allOutData)
            z2 = np.append(z,np.array(allOutArousal).reshape((len(allOutArousal),1)),axis=1)
            z2 = np.append(z2,np.array(allOutValence).reshape((len(allOutValence),1)),axis=1)
            z2 = np.append(z2,np.array(allOutDominance).reshape((len(allOutDominance),1)),axis=1)
            z2 = np.append(z2,np.array(allOutTrial).reshape((len(allOutTrial),1)),axis=1)
            z2 = np.append(z2,np.array(allOutParticipants).reshape((len(allOutParticipants),1)),axis=1)
            z2 = np.append(z2,np.array(allOutChannel).reshape((len(allOutChannel),1)),axis=1)
            z2 = np.append(z2,np.array(allOutTime).reshape((len(allOutTime),1)),axis=1)
            #print(z2)
            filename = './' + ('umapalltestorall' if not allChannel else 'tryallchannelonepart') + '/'+ base + '.csv'
            print(filename)
            pd.DataFrame(z2).to_csv(filename,index=False,header=['data_1','data_2','arousal','valence','dominance','trial','participant','channel','time'])
    else:
        # loading openbci data
        files = glob.glob(path)
        labelfiles = glob.glob(labels)

        for index in range(len(files)):
            print(files[index],labelfiles[index])
            outData,labels, outArousal, outValence, outDominance, outEventName, outTime, outChannels, outParticipants, outDimension = processData([files[index]],[labelfiles[index]],stft = True,seconds=seconds,isOpenBci=True,labelsMap=labelsMap,fmin=0,fmax=100)

            dimension = outData.shape[1]

            embedder,encoder = makeEEGNetwork((dimension,),n_components=3,og=True)  # 92

            embedder.fit_transform(outData)
            z = embedder.transform(outData)
            base = os.path.basename(files[index])
            filename = './' + base + f'processed{index}.csv'
            z2 = np.append(z,np.array(outArousal).reshape((len(outArousal),1)),axis=1)
            z2 = np.append(z2,np.array(outValence).reshape((len(outValence),1)),axis=1)
            z2 = np.append(z2,np.array(outDominance).reshape((len(outDominance),1)),axis=1)
            z2 = np.append(z2,np.array(outEventName).reshape((len(outEventName),1)),axis=1)
            z2 = np.append(z2,np.array(outParticipants).reshape((len(outParticipants),1)),axis=1)
            z2 = np.append(z2,np.array(outChannels).reshape((len(outChannels),1)),axis=1)
            z2 = np.append(z2,np.array(outTime).reshape((len(outTime),1)),axis=1)
            pd.DataFrame(z2).to_csv(filename,index=False,header=['data_1','data_2','data_3','arousal','valence','dominance','trial','participant','channel','time'])
        print(files,labels)
        pass



content=open('./chase/labelmap.yaml','r').read()
contentMap = yaml.load(content,Loader=yaml.FullLoader)


labelsMap = {}

for i in contentMap['IAPS'].keys():
    labelsMap[i] = contentMap['IAPS'][i]

for i in contentMap['groups'].keys():
    for j in contentMap['groups'][i].keys():
        if contentMap['groups'][i][j]['name'] in contentMap['files'].keys():
            labelsMap[f'{i}-{j}'] = contentMap['files'][contentMap['groups'][i][j]['name']]
        else:
            labelsMap[f'{i}-{j}'] = {'arousal':-1,'valence':-1,'dominance':-1}

print(labelsMap)
makeDataSet(path='./chase/*/Mash-Upsopenbci_eeg.csv',labels='./chase/*/Mash-UpsEvents.csv',seconds=1,is_deap=False,labelsMap=labelsMap)
#makeDataSet(path='./dataparticipantsdeap/data/*.bson',labels='./*',is_deap=True,allChannel=False,oneParticipant=True,seconds=5,labelsMap={})