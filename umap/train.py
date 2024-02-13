import glob
import pandas as pd
import numpy as np
import os
#from tpot import TPOTClassifier
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


import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.core.classes import Random
import weka.core.converters as converters
from weka.filters import Filter
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import FilteredClassifier
jvm.start(system_cp=True, packages=True)
#import lightgbm as lgb

#files = glob.glob('tryallchannelonepart/*.csv')
#files = glob.glob('testonechannelonepart/*.csv')
#files = glob.glob('ogwithouttrial/*.csv')
#files = glob.glob('ogumaponepartonechannel/*.csv')
#files = glob.glob('onechannelonepart/s1.bson_trainingdata_.csv')
#files = glob.glob('combineddall.csv')
#files = glob.glob('s20.bson_allchanneltrainingdata_.csv')
#files = glob.glob('onechannelparawithtrial/*.csv')

# use python 3 with this
def trainWeka(filename):
    print(filename)
    # Start the JVM
    
    
    # Load data
    #loader = Loader("weka.core.converters.CSVLoader")
    data = converters.load_any_file(filename,class_index="3")
    print(data.attribute_names())
    #data = loader.load_file('/mnt/h/school/affectivecomputing/EEG-Example-Workflow/umap/'+filename)

    numeric_to_nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "3-6"])
    numeric_to_nominal.inputformat(data)
    nomdata = numeric_to_nominal.filter(data)

    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "3,5-6"]) # 4-6
    remove.inputformat(nomdata)
    valence_filtered = remove.filter(nomdata)
    valence_filtered.class_index = 2

    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "4-6"]) # 4-6
    remove.inputformat(nomdata)
    arousal_filtered = remove.filter(nomdata)
    arousal_filtered.class_index = 2

    #for attribute in filtered.attributes():
    #    attribute_type = "Nominal" if attribute.is_nominal else "Numeric" if attribute.is_numeric else "Unknown"
    #    print(f"{attribute.name}: {attribute_type}")

    #print(filtered.attribute_names())

    # Initialize Random Forest classifier
    

    print('start')


    #print(len(filtered.attribute_names()))
    #print(data.class_index)
    #print(filtered.class_index)
    classifier = Classifier(classname="weka.classifiers.trees.REPTree")
    evl = Evaluation(arousal_filtered)
    evl.crossvalidate_model(classifier, arousal_filtered, 10, Random(1))
    arousal_correct = evl.percent_correct
    arousal_matrix = evl.confusion_matrix

    classifier = Classifier(classname="weka.classifiers.trees.REPTree")
    evl = Evaluation(valence_filtered)
    evl.crossvalidate_model(classifier, valence_filtered, 10, Random(1))
    valence_correct = evl.percent_correct
    valence_matrix = evl.confusion_matrix

    #print(evl.percent_correct)
    #print(evl.summary())
    #print(evl.confusion_matrix)
    #print(evl.class_details())

    # Train the classifier
    #classifier.build_classifier(data)

    print('done')

    return valence_correct,valence_matrix,arousal_correct,arousal_matrix



def processData(files,cls,features=['data_1','data_2','time'],output='results.csv',classify=True):
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
        #features = ['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time']
        #features = ['angle','magnitude', 'dist_center_1', 'dist_center_2']
        #features = ['data_1','data_2','time']
        data['trial']=data['trial'].apply(lambda x: int(x))
        data['participant']=data['participant'].apply(lambda x: str(x))
        basename = os.path.basename(file)
        newfilename = basename + 'processed' + '.csv'
        data.to_csv(newfilename,index=False)
        valence_correct,valence_matrix,arousal_correct,arousal_matrix = trainWeka(newfilename)
        df = pd.DataFrame(columns=['valence_correct_weka','arousal_correct_weka'])
        df['valence_correct_weka'] = [valence_correct]
        df['arousal_correct_weka'] = [arousal_correct]
        print(df)
        newfilename = basename + 'processed' + '.csv'
        data.to_csv(newfilename,index=False)
        df.to_csv(basename+'results.csv', index=False)
        print('saved' + newfilename)
        if classify:
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
    if classify:
        frame = pd.DataFrame(frame,columns=['partcipant','average_test_trial_accuracy','average_dominance_accuracy','average_valence_accuracy','average_arousal_accuracy'])
        frame.to_csv(output)
        print('done')


files = glob.glob('./alldata/*.csv')
processData(files,None,output=None,classify=False)


jvm.stop()

files = glob.glob('allbandsdata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='allresultsderivedfeatures.csv',features=['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time'])

files = glob.glob('betadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='betaresultsderivedfeatures.csv',features=['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time'])

files = glob.glob('alphadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='alpharesultsderivedfeatures.csv',features=['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time'])

files = glob.glob('deltadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='deltaresultsderivedfeatures.csv',features=['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time'])

files = glob.glob('gammadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='gammaresultsderivedfeatures.csv',features=['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time'])

files = glob.glob('thetadata/*.csv')
cls = ensemble.RandomForestClassifier(random_state=42)
processData(files,cls,output='thetaresultsderivedfeatures.csv',features=['angle','magnitude', 'dist_center_1', 'dist_center_2','time_x','time_y', 'time'])