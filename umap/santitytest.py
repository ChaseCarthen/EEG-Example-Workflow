from utils import makeEEGNetwork,plotData,plot_signal,plot_multichannel
from readOpenBCI import processData
import glob
files = glob.glob('/home/ccarthen/Downloads/Trial Data/**/*openbci_eeg.csv')
labelfiles = glob.glob('/home/ccarthen/Downloads/Trial Data/**/*Events.csv')
for index in range(len(files)):
    print(files[index],labelfiles[index])
    outData,labels,outChannels, outParticipants, outDimension = processData([files[index]],[labelfiles[index]],stft=False)
    print(outData.shape)
    print(labels)
    #print(outParticipants)
    print(outChannels)

    plot_multichannel(outData)

#embedder,encoder = makeEEGNetwork((outDimension,),og=True)  # 92
#points = embedder.fit_transform(outData)
#plotData('test',points)
