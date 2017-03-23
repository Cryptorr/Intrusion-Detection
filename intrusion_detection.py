import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

header = pd.read_table("kddcup.names.txt", header=None)
att_types = pd.read_table("training_attack_types.txt", sep=" ", header=None)

tr_raw = pd.read_csv("kddcup.data_10_percent_corrected", header=None)
test_raw = pd.read_csv("kddcup_testdata.corrected", header=None)

def preprocess(dat):
    dat.columns = header[0]
    att_types.columns = ["attack", "type"]
    dat["type"] = np.nan
    for i in range(0, len(att_types["attack"])):
        dat.loc[dat["attack"] == att_types.loc[i,].attack, "type"] = att_types.loc[i,].type
    dat.type = dat.type.fillna("unlisted")

    dat[dat.select_dtypes(include=['number']).columns] = preprocessing.scale(
        dat[dat.select_dtypes(include=['number']).columns])
    dat = pd.get_dummies(dat, columns=['protocol_type', 'service', 'flag', 'attack'])

    return dat


tr = preprocess(tr_raw)
test = preprocess(test_raw)  # actual test data (but labeled--don't cheat!)

# not occurring in training set but occuring in test set so we just add it
for i in np.setdiff1d(test.columns, tr.columns):
    tr[i] = 0
for i in np.setdiff1d(tr.columns, test.columns):
    test[i] = 0
test = test[tr.columns.tolist()]

tr_labels = tr["type"].values
tr_features = tr.drop(["type"], axis=1).values

test_labels = test["type"].values
test_features = test.drop(["type"], axis=1).values

clf = RandomForestClassifier(n_estimators=20, max_depth=None, random_state=0)
clf.fit(tr_features, tr_labels)
np.set_printoptions(suppress=True)
print(clf.score(test_features, test_labels))
types = tr['type'].unique()
print(types)
cm = confusion_matrix(test_labels, clf.predict(test_features), types)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
