import gdal
from osgeo import gdalconst
import numpy as np
import os
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
    results = np.zeros((row, col))

    for year in range(2004, 2016):
        file_name = 'H:\Taiwan_dynamic_data\data1\sus_class\RF' + str(year) + '.tif'
        data, a, b = read_tif(file_name)

        total_data[:,:,year-2004] = data

    #merge the high and very high areas, the low and very low areas
    total_data[total_data == 4] = 5
    #find the position of unchanged value
    for i in [1,2,3,5]:
        sus = np.full((row, col, 12), i)
        temp = total_data == sus
        susce_same = temp.sum(axis = 2)
        results[susce_same[:,:] == 12] = i

    results[data2004[:,:] == -9999] = -1
    TIF_functions.save_tif(results, "sus_change.tif", geotransform, proj)

