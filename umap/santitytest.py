from utils import makeEEGNetwork,plotData
from readOpenBCI import processData
import glob
files = glob.glob('/home/ccarthen/Downloads/Trial Data/T2-Mashup/*openbci_eeg.csv')
labels = glob.glob('/home/ccarthen/Downloads/Trial Data/T2-Mashup/*Events.csv')

outData,labels,outChannels, outParticipants, outDimension = processData(files,labels)
print(outData.shape)
print(labels)
#print(outParticipants)
print(outChannels)

embedder,encoder = makeEEGNetwork((outDimension,),og=True)  # 92
points = embedder.fit_transform(outData)
plotData('test',points)
