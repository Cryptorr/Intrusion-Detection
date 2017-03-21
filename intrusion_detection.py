import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

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
    dat.attack = dat.attack.astype('category')

    #scale
    nums = dat[dat.select_dtypes(include=['number']).columns]
    nums = pd.DataFrame(preprocessing.scale(nums))
    dat[dat.select_dtypes(include=['number']).columns] = nums
    return dat

tr = preprocess(tr_raw)
test = preprocess(test_raw)  # actual test data (but labeled--don't cheat!)

tr["dos_attack"] = np.where(tr.type == "dos",1,0)
tr["probe_attack"] = np.where(tr.type == "probe",1,0)
tr["r21_attack"] = np.where(tr.type == "r2l",1,0)
tr["u2r_attack"] = np.where(tr.type == "u2r",1,0)

columns = ["duration", "src_bytes", "dst_bytes", "num_failed_logins", "su_attempted",
            "root_shell", "num_file_creations", "count", "serror_rate"]

tr_labels = tr["attack"].values
tr_features = tr[list(columns)].values

test_labels = test["attack"].values
test_features = test[list(columns)].values

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(tr_features, tr_labels)
print clf.score(test_features, test_labels)