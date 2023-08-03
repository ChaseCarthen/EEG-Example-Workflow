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
from m5py import M5Prime
from sklearn.linear_model import RidgeClassifier
from lineartree import LinearTreeClassifier
from sklearn.linear_model import LinearRegression
from lineartree import LinearForestClassifier
from lineartree import LinearBoostClassifier
from sklearn.multiclass import OneVsRestClassifier


#files = glob.glob('tryallchannelonepart/*.csv')
#files = glob.glob('testonechannelonepart/*.csv')
#files = glob.glob('ogwithouttrial/*.csv')
#files = glob.glob('ogumaponepartonechannel/*.csv')
#files = glob.glob('onechannelonepart/s1.bson_trainingdata_.csv')
#files = glob.glob('combineddall.csv')
#files = glob.glob('s20.bson_allchanneltrainingdata_.csv')
#files = glob.glob('onechannelparawithtrial/*.csv')
files = glob.glob('umapalltestorall/*.csv')


mapping = {
    1: {
        "AVG_Valence": 6.8571,
        "AVG_Arousal": 5.8571,
        "AVG_Dominance": 6.0
    },
    2: {
        "AVG_Valence": 5.9286,
        "AVG_Arousal": 6.9286,
        "AVG_Dominance": 5.5
    },
    11: {
        "AVG_Valence": 7.1429,
        "AVG_Arousal": 4.8571,
        "AVG_Dominance": 5.2143
    },
    3: {
        "AVG_Valence": 6.9333,
        "AVG_Arousal": 6.4667,
        "AVG_Dominance": 5.8
    },
    12: {
        "AVG_Valence": 5.9286,
        "AVG_Arousal": 3.3571,
        "AVG_Dominance": 4.9286
    },
    13: {
        "AVG_Valence": 6.5714,
        "AVG_Arousal": 4.2143,
        "AVG_Dominance": 5.4286
    },
    14: {
        "AVG_Valence": 7.0667,
        "AVG_Arousal": 4.7333,
        "AVG_Dominance": 5.3333
    },
    15: {
        "AVG_Valence": 6.4667,
        "AVG_Arousal": 4.0,
        "AVG_Dominance": 4.9333
    },
    22: {
        "AVG_Valence": 4.2,
        "AVG_Arousal": 3.7333,
        "AVG_Dominance": 3.9333
    },
    23: {
        "AVG_Valence": 3.3333,
        "AVG_Arousal": 4.4667,
        "AVG_Dominance": 3.2
    },
    16: {
        "AVG_Valence": 5.1333,
        "AVG_Arousal": 2.4,
        "AVG_Dominance": 4.4667
    },
    24: {
        "AVG_Valence": 3.3333,
        "AVG_Arousal": 2.9333,
        "AVG_Dominance": 4.6667
    },
    25: {
        "AVG_Valence": 4.2,
        "AVG_Arousal": 3.6,
        "AVG_Dominance": 4.6
    },
    26: {
        "AVG_Valence": 4.2,
        "AVG_Arousal": 3.0,
        "AVG_Dominance": 3.3333
    },
    31: {
        "AVG_Valence": 3.6667,
        "AVG_Arousal": 5.4667,
        "AVG_Dominance": 4.6
    },
    32: {
        "AVG_Valence": 4.6667,
        "AVG_Arousal": 6.4,
        "AVG_Dominance": 4.9333
    },
    33: {
        "AVG_Valence": 3.9333,
        "AVG_Arousal": 6.1333,
        "AVG_Dominance": 5.5333
    },
    4: {
        "AVG_Valence": 7.0,
        "AVG_Arousal": 5.9333,
        "AVG_Dominance": 6.0667
    },
    5: {
        "AVG_Valence": 7.2,
        "AVG_Arousal": 7.3333,
        "AVG_Dominance": 6.5333
    },
    6: {
        "AVG_Valence": 6.1333,
        "AVG_Arousal": 6.2,
        "AVG_Dominance": 5.6
    },
    7: {
        "AVG_Valence": 6.6667,
        "AVG_Arousal": 6.4667,
        "AVG_Dominance": 5.9333
    },
    17: {
        "AVG_Valence": 6.0667,
        "AVG_Arousal": 3.0,
        "AVG_Dominance": 4.8
    },
    18: {
        "AVG_Valence": 7.1333,
        "AVG_Arousal": 3.8667,
        "AVG_Dominance": 5.0
    },
    19: {
        "AVG_Valence": 7.5333,
        "AVG_Arousal": 4.4667,
        "AVG_Dominance": 5.7333
    },
    8: {
        "AVG_Valence": 7.2667,
        "AVG_Arousal": 6.0667,
        "AVG_Dominance": 6.6667
    },
    20: {
        "AVG_Valence": 6.2667,
        "AVG_Arousal": 4.1333,
        "AVG_Dominance": 4.6667
    },
    9: {
        "AVG_Valence": 7.0667,
        "AVG_Arousal": 6.4,
        "AVG_Dominance": 6.8
    },
    27: {
        "AVG_Valence": 4.3333,
        "AVG_Arousal": 3.1333,
        "AVG_Dominance": 4.8
    },
    28: {
        "AVG_Valence": 3.25,
        "AVG_Arousal": 2.75,
        "AVG_Dominance": 2.9375
    },
    29: {
        "AVG_Valence": 3.4375,
        "AVG_Arousal": 3.625,
        "AVG_Dominance": 3.9375
    },
    30: {
        "AVG_Valence": 3.2,
        "AVG_Arousal": 3.6667,
        "AVG_Dominance": 3.2667
    },
    34: {
        "AVG_Valence": 4.7857,
        "AVG_Arousal": 6.3571,
        "AVG_Dominance": 4.9286
    },
    21: {
        "AVG_Valence": 4.1429,
        "AVG_Arousal": 4.2143,
        "AVG_Dominance": 4.0
    },
    35: {
        "AVG_Valence": 3.5333,
        "AVG_Arousal": 6.3333,
        "AVG_Dominance": 4.9333
    },
    36: {
        "AVG_Valence": 4.9333,
        "AVG_Arousal": 7.2667,
        "AVG_Dominance": 7.0667
    },
    37: {
        "AVG_Valence": 3.2667,
        "AVG_Arousal": 5.8667,
        "AVG_Dominance": 5.5333
    },
    38: {
        "AVG_Valence": 3.2667,
        "AVG_Arousal": 5.3333,
        "AVG_Dominance": 5.6667
    },
    39: {
        "AVG_Valence": 3.4667,
        "AVG_Arousal": 5.3333,
        "AVG_Dominance": 5.2667
    },
    10: {
        "AVG_Valence": 5.8667,
        "AVG_Arousal": 7.0667,
        "AVG_Dominance": 7.0667
    },
    40: {
        "AVG_Valence": 3.7333,
        "AVG_Arousal": 5.7333,
        "AVG_Dominance": 5.5333
    }
}


for file in files:
    #cls = ensemble.RandomForestClassifier(random_state=42)
    #cls = tree.ExtraTreeClassifier(random_state=42,criterion='entropy',min_samples_split=2,min_samples_leaf=1)
    #cls = svm.LinearSVC(random_state=42)
    #cls = neural_network.MLPClassifier(random_state=42)
    #cls = semi_supervised.SelfTrainingClassifier(cls)
    #cls = ensemble.GradientBoostingClassifier(random_state=42)
    #cls = M5Prime(random_state=42)
    #cls = OneVsRestClassifier(LinearForestClassifier(base_estimator=LinearRegression(),max_features='sqrt',random_state=42))
    cls = LinearTreeClassifier(base_estimator=linear_model.LogisticRegression(random_state=42),max_depth=5)
    #cls = Orange.classification.TreeLearner()
    data = pd.read_csv(file)
    #data = data.drop(index=np.where(data['participant']==23.0)[0])
    #print(data.keys())
    #print('===============')
    #print(data[['trial','arousal','valence','dominance']].value_counts())
    #print(data[['trial','valence']].value_counts())
    #print(data[['trial','dominance']].value_counts())
    #print('---------------')

    #valence_labels = np.array([ mapping[int(i)+1]['AVG_Valence'] for i in data['trial']]) > 5
    #arousal_labels = np.array([ mapping[int(i)+1]['AVG_Arousal'] for i in data['trial']]) > 5
    #dominance_labels = np.array([ mapping[int(i)+1]['AVG_Dominance'] for i in data['trial']]) > 5
    #data['arousal'] = data['arousal'] > 5.0
    #data['valence'] = data['valence'] > 5.0
    #data['dominance'] = data['dominance'] > 5.0

    #print(data['arousal'].min(),data['arousal'].max(),data['arousal'].min() + (data['arousal'].max() - data['arousal'].min()) / 2)
    #data['arousal'] = data['arousal'] > data['arousal'].min() + (data['arousal'].max() - data['arousal'].min()) / 2
    #data['valence'] = data['valence'] > data['arousal'].min() + (data['valence'].max() - data['valence'].min()) / 2
    #data['dominance'] = data['dominance'] > data['arousal'].min() + (data['dominance'].max() - data['dominance'].min()) / 2
    #cls = TPOTClassifier()
    # split the data into 10 folds using scikit learn

    
    #cls = linear_model.LogisticRegression(random_state=42,max_iter=1000)
    #cls.fit(data[['data_1','data_2']], data['arousal'])

    #data['data_1'] = (data['data_1'] - data['data_1'].mean()) / data['data_1'].std()
    #data['data_2'] = (data['data_2'] - data['data_2'].mean()) / data['data_2'].std()

    #data= data.groupby(['trial','arousal','dominance','valence']).agg(data_1=('data_1','mean'),data_2=('data_2','mean')).reset_index()
    #input()
    data['data_1x2'] = data['data_1'] * data['data_2']
    data['dist_center_1'] = (data['data_1'] - data['data_1'].mean()) / data['data_1'].std()
    data['dist_center_2'] = (data['data_2'] - data['data_2'].mean()) / data['data_1'].std()
    
    data['magnitude'] = np.sqrt(data['dist_center_1']**2 + data['dist_center_2']**2) ** 2 * np.pi
    data['magnitude'] = (data['magnitude'] - data['magnitude'].mean()) / data['magnitude'].std()
    data['try_1'] = np.exp(data['data_1']) / (np.pi * 2)
    data['try_2'] = np.exp(data['data_2']) / (np.pi * 2)
    data['angle'] = np.arctan(data['data_2'] / (data['data_1'] + .000000001))
    #data['log_data_1'] = np.exp(data['data_1'])
    #data['log_data_2'] = np.exp(data['data_2'])
    data['x'] = np.cos(data['angle'])
    data['y'] = np.sin(data['angle'])

    data['x_y'] = data['x'] * data['y']

    data['time_x'] = np.cos(data['time']/60 * np.pi * 2)
    data['time_y'] = np.sin(data['time']/60 * np.pi * 2)

    data['dist'] = data['dist_center_1'] * data['dist_center_2']
    #data['data_1'] = (data['data_1'] - data['data_1'].mean()) / data['data_1'].std()
    #data["data_2"] = (data["data_2"] - data["data_2"].mean()) / data["data_2"].std()
    #print(data['try_2'])
    #data['data_1x2'] = (data['data_1x2'] - data['data_1x2'].mean()) / data['data_1x2'].std()
    #features = ['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time']
    #features = ['angle','magnitude', 'dist_center_1', 'dist_center_2']
    features = ['data_1','data_2','time','participant']
    data['trial']=data['trial'].apply(lambda x: int(x))
    data['time']=data['time'].apply(lambda x: int(x))
    data['participant']=data['participant'].apply(lambda x: int(x))
    #print(cls.predict(data[features])[0])
    #cross_val_score(cls,data[features], data['trial'], cv=4)
    scores = cross_validate(cls, data[features], data['trial'], cv=4, scoring='accuracy', return_train_score=False, return_estimator=False,verbose=True)
    print(file,scores['test_score'].mean())
    #print(np.where(scores['test_score'] == scores['test_score'].max()))
    #cls = scores['estimator'][np.where(scores['test_score'] == scores['test_score'].max())[0][0]]
    #data['trial'] = cls.predict(data[features])
    #data['trial'].apply(lambda x: int(x))
    #cls = ensemble.RandomForestClassifier(random_state=42)
    #print(cross_val_score(cls,data[features], data['arousal']).mean())
    #features.append('trial')
    #print(cross_val_score(cls,data[features], data['arousal']).mean())
    #print(cls.predict(data[features])[0])