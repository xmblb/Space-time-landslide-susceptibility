import gdal
from osgeo import gdalconst
import numpy as np
from sklearn.linear_model import LinearRegression
import gc
from common_func import TIF_functions


def read_tif(file):
    dem = gdal.Open(file)
    col = dem.RasterXSize
    row = dem.RasterYSize
    band = dem.RasterCount
    geotransform = dem.GetGeoTransform()
    proj = dem.GetProjection()
    data = np.zeros([row, col, band])
    for i in range(band):
        sing_band = dem.GetRasterBand(i+1)
        data[:,:,i] = sing_band.ReadAsArray()
    data = data.reshape(row, col)
    return data, geotransform, proj



if __name__ == '__main__':

    data2004, geotransform, proj = read_tif('Mask.tif')
    row, col = data2004.shape
    total_data = np.zeros((row, col, 12))
    # print(total_data[:,:,0].shape)
    # results = np.zeros((row, col))

    for year in range(2004, 2016):
        file_name = 'H:\Taiwan_dynamic_data\data1\RF' + str(year) + '.tif'

        data, a, b = read_tif(file_name)

        total_data[:,:,year-2004] = data

    mean_sus = total_data.mean(axis=2)

    mean_sus[data2004[:,:] == -9999] = -1
    TIF_functions.save_tif(mean_sus, "mean_sus.tif", geotransform, proj)


    max_sus = total_data.max(axis=2)
    min_sus = total_data.min(axis=2)

    max_sus[data2004[:,:] == -9999] = -1
    min_sus[data2004[:, :] == -9999] = -1
    TIF_functions.save_tif(max_sus, "max_sus.tif", geotransform, proj)
    TIF_functions.save_tif(min_sus, "min_sus.tif", geotransform, proj)

