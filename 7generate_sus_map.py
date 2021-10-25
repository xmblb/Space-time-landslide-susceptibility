import gdal
import h5py
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import gc
from sklearn import metrics
from common_func import evaluate_method,TIF_functions
import time
np.random.seed(1)

def get_sus_map(imgs,mask, model):
    # lenth=len(imgs)
    row=mask.shape[0]
    col=mask.shape[1]

    mask = np.reshape(mask,[row*col,])

    proba=model.predict_proba(imgs)
    proba=proba[:,1]
    mask[mask[:]==0] = proba
    mask=np.reshape(mask,[row,col])
    return mask


if __name__ == '__main__':
    start = time.time()
    ratio = 1.3
    year = 2010
    n_est = 800

    # load total data
    f = h5py.File('total_data_vector'+str(year)+'.h5', 'a')
    total_data = f['data']
    # total_data = np.array(f['data'])
    print(total_data.shape)

    #remove the redundant features
    retained_features_index =   np.load('retained_features'+ str(year) + '.npy')

    landslide_train_name = 'train_landslide'+ str(year) + '.npy'
    nonlandslide_total_train_name = 'train_nonlandslide'+ str(year) + '.npy'

    # 读取滑坡和非滑坡数据
    landslide_train = np.load(landslide_train_name)
    nonlandslide_total_train = np.load(nonlandslide_total_train_name)
    nonlandslide_train_value = np.random.choice(range(len(nonlandslide_total_train)), int(len(landslide_train)*ratio), replace=False)
    nonlandslide_train = np.array([nonlandslide_total_train[a] for a in nonlandslide_train_value])

    train_data_x = np.concatenate((landslide_train, nonlandslide_train), axis=0)
    train_data_x = train_data_x[:,retained_features_index]
    train_data_y = np.append(np.ones(len(landslide_train),dtype=int),np.zeros(int(len(landslide_train)*ratio),dtype=int))

    # build RF model
    model = RandomForestClassifier(n_estimators=n_est)
    model.fit(train_data_x, train_data_y)

    #generate susceptibility map
    col, row, geotransform, proj, mask = TIF_functions.read_tif('Mask.tif')
    mask = np.reshape(mask, (row, col))
    total_prob = get_sus_map(total_data, mask, model)
    total_prob[mask[:, :] == -9999] = -1
    TIF_functions.save_tif(total_prob, "RF"+str(year)+".tif", geotransform, proj)

    del total_data, total_prob
    gc.collect()
    f.close()

    end = time.time()
    print(end-start)