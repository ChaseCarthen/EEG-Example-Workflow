from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold,cross_val_score,cross_validate
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np 
df = pd.read_csv('./train.csv')
target = df[["valence","arousal"]].to_numpy()
trainingColumns = np.array([ (column if 'power' in column else None) for column in df.columns])
trainingColumns = trainingColumns[trainingColumns != None]
training = df[trainingColumns].to_numpy()
df = pd.read_csv('./test.csv')
testing = df[trainingColumns] 
#print(training)

clf = RandomForestClassifier(random_state=0)

kf = KFold(n_splits=5,shuffle=True)
scores = cross_validate(clf, training, target, cv=kf, return_estimator=True)

print(scores['estimator'][0].predict(training))
print(scores['estimator'][0].predict(testing))
#for fold, score in enumerate(scores):
#    print(f"Fold {fold+1}: {score}")

#print(clf.fit(training, target))
#print(clf.predict(training))
#print(clf.predict(testing))