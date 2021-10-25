import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLogisticRegression,RandomizedLasso
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tools.tools import add_constant

def remove_features(original_data, index):
    '''remove redundant features'''
    output = np.zeros((len(original_data), len(index)))
    for i in range(len(index)):
        output[:,i] = original_data[:,index[i]]
    return output

if __name__ == '__main__':


    year = 2015
    landslide_train = np.load('train_landslide'+str(year) + '.npy')
    nonlandslide_total_train = np.load('train_nonlandslide'+str(year) + '.npy')

    nonlandslide_index = np.random.choice(range(len(nonlandslide_total_train)), len(landslide_train), replace=False)
    nonlandslide_train = np.array([nonlandslide_total_train[a] for a in nonlandslide_index])

    train_x = np.concatenate((landslide_train, nonlandslide_train), axis=0)
    train_y = np.append(np.ones(len(landslide_train),dtype=int),np.zeros(len(nonlandslide_train),dtype=int))
    # train_x[train_x[:,:] == -9999] = 1
    print(train_x.min(), train_x.max())
    # multicollinearity
    train_y_coll = np.reshape(train_y, (len(train_y), 1))
    mutli_coll_data = np.concatenate((train_x,train_y_coll), axis=1)
    # np.savetxt('data.txt', mutli_coll_data)
    mutli_coll_data = add_constant(mutli_coll_data)

    vif=[]
    no_coll_index = []
    for i in range(mutli_coll_data.shape[1]-1):
        value = variance_inflation_factor(mutli_coll_data,i)
        vif.append(round(value, 4))
        if value < 10:
            no_coll_index.append(i-1)

    np.savetxt('vif'+ str(year) + '.txt', vif)
    print(no_coll_index)
    train_x_new = remove_features(train_x, no_coll_index)

    names = range(train_x_new.shape[1])
    # rlasso = RandomizedLogisticRegression()
    rlasso = RandomForestClassifier(n_estimators=500)
    rlasso.fit(train_x_new, train_y)
    print(sorted(zip(map(lambda x: round(x, 4), rlasso.feature_importances_), names), reverse=True))

    importance = np.array([round(x, 4) for x in rlasso.feature_importances_])
    np.savetxt('importance' + str(year) + '.txt', importance)

    index = []
    for i in range(len(rlasso.feature_importances_)):
        if rlasso.feature_importances_[i] >= 0.01:
            index.append(no_coll_index[i])
    print(len(index), index)
    # np.save('retained_features_vif' + str(year) + '.npy', np.array(no_coll_index))
    np.save('retained_features'+ str(year) + '.npy', np.array(index))

