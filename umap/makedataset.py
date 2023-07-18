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

files = glob.glob('/mnt/data3/schooldata/bsonfiles/*.bson')

allOutData=None
allOutParticipants=None 
allOutChannel=None
allOutArousal=None
allOutValence=None
allOutDominance=None
allOutTrial=None
embedder = None
encoder = None
count = 0
for file in files:
    
    base = os.path.basename(file)
    participant = int(re.findall('[0-9]+',base)[0])
    #print(os.path.basename(file))
    outData,outParticipants,outChannel,outArousal,outValence,outDominance,outTrial = loadDeapFile(file,partcipantNumber=participant,allChannel=False)

    if allOutData is not None:
        allOutData = np.append(allOutData,outData,axis=0)
        allOutParticipants = np.append(allOutParticipants,outParticipants,axis=0)
        allOutChannel = np.append(allOutChannel,outChannel,axis=0)
        allOutDominance = np.append(allOutDominance,outDominance,axis=0)
        allOutArousal = np.append(allOutArousal,outArousal,axis=0)
        allOutValence = np.append(allOutValence,outValence,axis=0)
        allOutTrial = np.append(allOutTrial,outTrial,axis=0)
    else:
        allOutData = outData
        allOutParticipants = outParticipants
        allOutChannel = outChannel
        allOutDominance = outDominance
        allOutArousal = outArousal
        allOutValence = outValence
        allOutTrial = outTrial

    print(outData.shape)

    print(outArousal.shape)
    #input()
    #input()
    dimension = outData.shape[1]
    if embedder is None and encoder is None:
        embedder,encoder = makeEEGNetwork((dimension,))  # 92
    if count == 10:
        embedder.fit_transform(allOutData)
    count += 1

z = encoder.predict(allOutData)
z2 = np.append(z,np.array(allOutArousal).reshape((len(allOutArousal),1)),axis=1)
z2 = np.append(z2,np.array(allOutValence).reshape((len(allOutValence),1)),axis=1)
z2 = np.append(z2,np.array(allOutDominance).reshape((len(allOutDominance),1)),axis=1)
z2 = np.append(z2,np.array(allOutTrial).reshape((len(allOutTrial),1)),axis=1)
z2 = np.append(z2,np.array(allOutParticipants).reshape((len(allOutParticipants),1)),axis=1)
z2 = np.append(z2,np.array(allOutChannel).reshape((len(allOutChannel),1)),axis=1)
pd.DataFrame(z2).to_csv('combineddall.csv',index=False,header=['data_1','data_2','arousal','valence','dominance','trial','participant','channel'])
