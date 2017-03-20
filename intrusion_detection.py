import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing

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