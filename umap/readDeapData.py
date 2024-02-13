import bson 
import glob
import numpy as np 
from utils import processSignal


def loadDeapFile(filename,partcipantNumber=0,allChannel=False,fmin=13,fmax=30,samplerate=128,seconds=5,useGroupsRating=False):
    data = open(filename,'rb').read()
    data = bson.loads(data)
    outData = None
    outParticipants = []
    outChannel = []
    outArousal = []
    outValence = []
    outDominance = []
    outTrial = []
    outTime = []
    #print(len(data['channel']))
    # iterate across 40 trials
    for trial in range(40):
        # iterate across 32
        channelData = None
        for channel in range(32):
            
            # create stfts and labels
            signalData = np.array(data['trial'][str(trial)]['channels'][channel]['data'])[:128*60]
            processedData,time,fmaxindex,fminindex = processSignal(signalData,numChannels=1,seconds=128*seconds,samplerate=128,fmin=fmin,fmax=fmax)
            if channelData is None:
                channelData = processedData
            else:
                channelData = np.append(channelData,processedData,axis= (0 if not allChannel else 1))
            if not allChannel:
                for t in time:
                    outParticipants.append(partcipantNumber)
                    outChannel.append(channel)
                    outTrial.append(trial)
                    if not useGroupsRating:
                        outArousal.append(data['trial'][str(trial)]['channels'][0]['arousal_label'])
                        outValence.append(data['trial'][str(trial)]['channels'][0]['valence_label'])
                    else:
                        outArousal.append(data['trial'][str(trial)]['channels'][0]['group_arousal_label'])
                        outValence.append(data['trial'][str(trial)]['channels'][0]['group_valence_label'])
                    outDominance.append(data['trial'][str(trial)]['channels'][0]['dominance_label'])
                    outTime.append(t)
        if allChannel:
            for t in time:
                outParticipants.append(partcipantNumber)
                outChannel.append(channel)
                outTrial.append(trial)
                outArousal.append(data['trial'][str(trial)]['channels'][0]['arousal_label'])
                outValence.append(data['trial'][str(trial)]['channels'][0]['valence_label'])
                outDominance.append(data['trial'][str(trial)]['channels'][0]['dominance_label'])
                outTime.append(t)

        #print(channelData)
        #print(channelData.shape)
        #print(len(time))
        if outData is None:
            outData = channelData
        else:
            outData = np.append(outData,channelData,axis=0)
        #input()
    
    return np.array(outData),np.array(outParticipants),np.array(outChannel),np.array(outArousal),np.array(outValence),np.array(outDominance),np.array(outTrial),np.array(outTime)






