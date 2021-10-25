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

    # load total data
    f = h5py.File('total_data2015.h5', 'a')
    total_data = f['data']
    # total_data = np.array(f['data'])
    print(total_data.shape)

    col, row, geotransform, proj, mask = TIF_functions.read_tif('Mask.tif')
    #remove the redundant features from total data
    retained_features_index = np.load('retained_features2015.npy')
    imgs = total_data[:,:,retained_features_index]
    print(imgs.shape)
    del total_data
    gc.collect()

    imgs = np.reshape(imgs, (row*col, imgs.shape[2]))
    print(imgs.shape)

    if not os.path.exists('total_data_fs2015.h5'):
        with h5py.File('total_data_fs2015.h5') as f1:
            f1.create_dataset('data', data=imgs, compression='gzip', compression_opts=9, dtype='float32')

    end = time.time()
    f.close()
    del imgs
    gc.collect()
    print(end-start)