import glob
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.model_selection
from sklearn.model_selection import cross_val_score,cross_validate
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.neural_network as neural_network
import sklearn.semi_supervised as semi_supervised
import math
#from m5py import M5Prime
from sklearn.linear_model import RidgeClassifier
#from lineartree import LinearTreeClassifier
from sklearn.linear_model import LinearRegression
#from lineartree import LinearForestClassifier
#from lineartree import LinearBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

#import lightgbm as lgb

#files = glob.glob('tryallchannelonepart/*.csv')
#files = glob.glob('testonechannelonepart/*.csv')
#files = glob.glob('ogwithouttrial/*.csv')
#files = glob.glob('ogumaponepartonechannel/*.csv')
#files = glob.glob('onechannelonepart/s1.bson_trainingdata_.csv')
#files = glob.glob('combineddall.csv')
#files = glob.glob('s20.bson_allchanneltrainingdata_.csv')
#files = glob.glob('onechannelparawithtrial/*.csv')


def processData(files,cls,features=['data_1','data_2','time'],output='results.csv'):
    frame = []
    for file in files:
        cls = ensemble.RandomForestClassifier(random_state=42)
        data = pd.read_csv(file)

        data['arousal'] = data['arousal'] > 5.0
        data['valence'] = data['valence'] > 5.0
        data['dominance'] = data['dominance'] > 5.0
        data['data_1x2'] = data['data_1'] * data['data_2']
        data['dist_center_1'] = (data['data_1'] - data['data_1'].mean()) / data['data_1'].std()
        data['dist_center_2'] = (data['data_2'] - data['data_2'].mean()) / data['data_1'].std()
        data['magnitude'] = np.sqrt(data['dist_center_1']**2 + data['dist_center_2']**2) ** 2 * np.pi
        data['magnitude'] = (data['magnitude'] - data['magnitude'].mean()) / data['magnitude'].std()
        data['try_1'] = np.exp(data['data_1']) / (np.pi * 2)
        data['try_2'] = np.exp(data['data_2']) / (np.pi * 2)
        data['angle'] = np.arctan(data['data_2'] / (data['data_1'] + .000000001))
        data['x'] = np.cos(data['angle'])
        data['y'] = np.sin(data['angle'])
        data['x_y'] = data['x'] * data['y']
        data['time_x'] = np.cos(data['time']/60 * np.pi * 2)
        data['time_y'] = np.sin(data['time']/60 * np.pi * 2)
        data['dist'] = data['dist_center_1'] * data['dist_center_2']
        features = ['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time']
        #features = ['angle','magnitude', 'dist_center_1', 'dist_center_2']
        #features = ['data_1','data_2','time']
        data['trial']=data['trial'].apply(lambda x: int(x))
        data['participant']=data['participant'].apply(lambda x: str(x))
        scores = cross_validate(cls, data[features], data['trial'], cv=4, scoring='accuracy', return_train_score=False, return_estimator=False)
        trialscore = scores['test_score'].mean()

        scores = cross_validate(cls, data[features], data['arousal'], cv=4, scoring='accuracy', return_train_score=False, return_estimator=False)
        arousalscore = scores['test_score'].mean()

        scores = cross_validate(cls, data[features], data['dominance'], cv=4, scoring='accuracy', return_train_score=False, return_estimator=False)
        dominancescore = scores['test_score'].mean()

        scores = cross_validate(cls, data[features], data['valence'], cv=4, scoring='accuracy', return_train_score=False, return_estimator=False)
        valencescore = scores['test_score'].mean()

        frame.append([data['participant'][0],trialscore,dominancescore,valencescore,arousalscore])
        print(data['participant'][0])
    frame = pd.DataFrame(frame,columns=['partcipant','average_test_trial_accuracy','average_dominance_accuracy','average_valence_accuracy','average_arousal_accuracy'])
    frame.to_csv(output)
    print('done')


files = glob.glob('allbandsdata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='allresults.csv')

files = glob.glob('betadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='betaresults.csv')

files = glob.glob('alphadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='alpharesults.csv')

files = glob.glob('deltadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='deltaresults.csv')

files = glob.glob('gammadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='gammaresults.csv')

files = glob.glob('thetadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='gammaresults.csv')