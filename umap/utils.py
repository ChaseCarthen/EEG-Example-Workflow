from scipy import signal
import numpy as np
from umap.parametric_umap import ParametricUMAP
from umap import UMAP
import tensorflow as tf
import matplotlib.pyplot as plt
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

def makeEEGNetwork(inputDims,n_components=2,og=True):
    if og:
        embedder = UMAP(n_components=2,metric='correlation')
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

def plotData(title, data, labels=[], cmap='tab20', showLabelText=False, save=False, saveName='plot.png',showPlot=True):
    fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
    if len(labels) > 0:
        sc = ax.scatter(
            data[:, 0],
            data[:, 1],
            c=labels,
            s=4,
            alpha=0.5,
            rasterized=True
        )
    else:
        sc = ax.scatter(
            data[:, 0],
            data[:, 1],
            s=4,
            alpha=0.5,
            rasterized=True
        )
    showLabelText = showLabelText and len(labels) > 0
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
    if showPlot:
        plt.show()

def plot_signal(signal, x_axis=None, title="Signal Plot", xlabel="Time", ylabel="Amplitude"):
    """
    Plots a given signal.

    Parameters:
    - signal: List or array of signal values.
    - x_axis: Optional list or array for the x-axis (e.g., time values).
    - title: Title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """

    plt.figure(figsize=(10, 5))
    if x_axis:
        plt.plot(x_axis, signal)
    else:
        plt.plot(signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_multichannel(signals, channel_names=None, title="Multichannel Signal Plot", xlabel="Time", ylabel="Amplitude"):
    """
    Plots signals from multiple channels.

    Parameters:
    - signals: 2D list or array where each row is a channel.
    - channel_names: Optional list of channel names.
    - title, xlabel, ylabel: Plot labels.
    """

    n_channels = len(signals)
    if channel_names == None:
        channel_names = [str(i) for i in range(n_channels)] 
    if channel_names and len(channel_names) != n_channels:
        raise ValueError("Mismatch between number of signals and channel names.")

    plt.figure(figsize=(10, 5 * n_channels))
    for idx, channel in enumerate(signals):
        plt.subplot(n_channels, 1, idx + 1)
        plt.plot(channel)
        if channel_names:
            plt.title(channel_names[idx])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
