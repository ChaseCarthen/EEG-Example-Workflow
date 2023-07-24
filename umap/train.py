import glob
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
import sklearn.ensemble as ensemble
import sklearn.model_selection
from sklearn.model_selection import cross_val_score
import math
#files = glob.glob('allchannelonepart/*.csv')
#files = glob.glob('onechannelonepart/*.csv')
files = glob.glob('onechannelonepart/s1.bson_trainingdata_.csv')
files = glob.glob('combineddall.csv')
for file in files:
    data = pd.read_csv(file)
    #print(data.keys())
    print('===============')
    print(data[['trial','arousal','valence','dominance']].value_counts())
    #print(data[['trial','valence']].value_counts())
    #print(data[['trial','dominance']].value_counts())
    print('---------------')

    data['arousal'] = round(data['arousal']) > 5
    data['valence'] = round(data['valence']) > 5
    data['dominance'] = round(data['dominance']) > 5

    #cls = TPOTClassifier()
    # split the data into 10 folds using scikit learn

    cls = ensemble.RandomForestClassifier()
    #cls.fit(data[['data_1','data_2']], data['arousal'])

    print(file,cross_val_score(cls, data[['data_1','data_2']], data['arousal'], cv=10))