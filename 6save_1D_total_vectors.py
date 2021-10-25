import gdal
import h5py
import numpy as np
import os
import gc
from common_func import evaluate_method,TIF_functions
import time
np.random.seed(1)

if __name__ == '__main__':
    start = time.time()

    year = 2007
    # load total data
    f = h5py.File('total_data'+str(year)+'.h5', 'a')
    total_data = f['data']
    # total_data = np.array(f['data'])3.52
    print(total_data.shape)
    retained_features_index =   np.load('retained_features'+ str(year) + '.npy')
    total_data = total_data[:,:,retained_features_index]

    col, row, geotransform, proj, mask = TIF_functions.read_tif('Mask.tif')
    mask = np.reshape(mask, (row * col, ))
    total_data = np.reshape(total_data, (row * col, total_data.shape[2]))
    # img = np.reshape( f['data'], [row * col,  f['data'].shape[2]])[mask == 0,:]
    # del total_data
    # gc.collect()
    img = np.zeros((47772469, total_data.shape[1]))
    for i in range(total_data.shape[1]):
        print(i)
        # print(total_data[:,i].shape)
        img[:,i]=total_data[:,i][mask==0]

    # total_data = f['data'][mask == 0, :]
    print(img.shape)
    # print(img)
    file_name = 'total_data_vector'+str(year)+'.h5'
    if not os.path.exists(file_name):
        with h5py.File(file_name) as f1:
            f1.create_dataset('data', data=img, compression='gzip', compression_opts=9, dtype='float32')

    end = time.time()
    f.close()
    del total_data
    gc.collect()
    print(end-start)