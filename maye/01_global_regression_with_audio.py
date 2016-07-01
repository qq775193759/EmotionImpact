import numpy as np
from math import sqrt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import scale

audio_features = {}
with open('mfcc.csv', 'r') as fin:
    for line in fin:
        l = line.strip().split('\t')
        id = int(l[0])
        f = [float(x) for x in l[1:]]
        audio_features[id] = f

a_features = {}
v_features = {}
with open('ACCEDEfeaturesArousal_TAC2015.txt', 'r') as fin:
    for line in fin:
        l = line.strip().split('\t')
        if l[0] == 'id':
            continue
        id = int(l[0])
        f = [float(x) for x in l[2:]]
        a_features[id] = f

with open('ACCEDEfeaturesValence_TAC2015.txt', 'r') as fin:
    for line in fin:
        l = line.strip().split('\t')
        if l[0] == 'id':
            continue
        id = int(l[0])
        f = [float(x) for x in l[2:]]
        v_features[id] = f

a_map = {}
v_map = {}

with open('ACCEDEranking.txt', 'r') as fin:
    for line in fin:
        l = line.strip().split('\t')
        if l[0] == 'id':
            continue
        id = int(l[0])
        a = float(l[5])
        v = float(l[4])
        a_map[id] = a
        v_map[id] = v

id2set = {}
with open('../cv_results/CVsets.txt', "r") as fin:
    for line in fin:
        l = line.strip().split('\t')
        id2set[int(l[0])] = int(l[1])

scores_r = [0, 0, 0, 0, 0]
scores_mse = [0, 0, 0, 0, 0]

for current in range(5):
    X_train = []
    Y_train = []
    X_predict = []
    Y_predict = []
    for i in id2set.items():
        id = i[0]
        s = i[1]
        if s == current:
            # test
            X_predict.append(v_features[id] + audio_features[id])
            Y_predict.append(v_map[id])
        else:
            # train
            X_train.append(v_features[id] + audio_features[id])
            Y_train.append(v_map[id])
    assert len(X_train) == len(Y_train)
    assert len(X_predict) == len(Y_predict)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_predict = np.array(X_predict)
    Y_predict = np.array(Y_predict)

    # X_train = scale(X_train, axis=0)
    # X_predict = scale(X_predict, axis=0)

    regr = AdaBoostRegressor(n_estimators=150, learning_rate=0.1)
    # regr = SVR(C=0.02, epsilon=0.5)

    regr.fit(X_train, Y_train)
    y = regr.predict(X_predict)
    scores_mse[current] = mean_squared_error(Y_predict, y)
    scores_r[current] = pearsonr(Y_predict, y)[0]

# print(sum(scores) / 5)
print(scores_mse, sum(scores_mse) / 5)
print(scores_r, sum(scores_r) / 5)

