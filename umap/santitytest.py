from readOpenBCI import processData
import glob
files = glob.glob('/home/ccarthen/Downloads/Trial Data/Deep Fakes/*/*openbci_eeg.csv')
labels = glob.glob('/home/ccarthen/Downloads/Trial Data/Deep Fakes/*/*Events.csv')

outData,labels,outChannels, outParticipants, outDimension = processData(files,labels)
print(outData.shape)
print(labels)
#print(outParticipants)
print(outChannels)

