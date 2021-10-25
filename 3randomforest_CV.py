import gdal
from osgeo import gdalconst
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from common_func import TIF_functions
# from sklearn import metrics
# from common_func import evaluate_method
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
np.random.seed(1)

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
    return data

def save_tif(data, ouput_file, num, geotransform, proj):
    row,col, bands = data.shape
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(ouput_file, col, row, num, gdal.GDT_Float64)
    outRaster.SetGeoTransform(geotransform)
    outRaster.SetProjection(proj)
    for i in range(num):
        outband = outRaster.GetRasterBand(i+1)
        # mid_data = data[:,:,i]
        # mid_data = mid_data.reshape((row, col))
        outband.WriteArray(data[:,:,i])
        outband.SetNoDataValue(-1)
    outRaster.FlushCache()
    del outRaster

# resample function
def Resample(input_file, reference_file, output_file):
    #读取需要重采样栅格信息
    in_raster = gdal.Open(input_file)
    in_geotransform = in_raster.GetGeoTransform()
    in_proj = in_raster.GetProjection()

    #读取重采样参数
    reference_raster = gdal.Open(reference_file)
    refer_geotransform = reference_raster.GetGeoTransform()
    refer_proj = reference_raster.GetProjection()

    # 创建重采样输出栅格
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(output_file, 7326, 13709, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform(refer_geotransform)
    outRaster.SetProjection(refer_proj)

    gdal.ReprojectImage(in_raster, outRaster, in_proj, refer_proj, gdalconst.GRA_NearestNeighbour)
    del outRaster

def remove_features(original_data, index):
    '''remove redundant features'''
    output = np.zeros((len(original_data), len(index)))
    for i in range(len(index)):
        output[:,i] = original_data[:,index[i]]
    return output

if __name__ == '__main__':
    ratios = [1.4, 1.4, 1.3, 1.3, 1.4, 1.4, 1.4, 1.4, 1.5]
    years = [2007,2008,2009,2010,2011,2012,2013,2014,2015]
    for i in range(9):

        ratio = ratios[i]
        year = years[i]
        removed_features_index = np.load('retained_features'+ str(year) + '.npy')
        landslide_train_name = 'train_landslide'+ str(year) + '.npy'
        nonlandslide_total_train_name = 'train_nonlandslide'+ str(year) + '.npy'
        landslide_test_name = 'test_landslide'+ str(year) + '.npy'
        nonlandslide_total_test_name = 'test_nonlandslide'+ str(year) + '.npy'
        para_space = {'n_estimators': range(100, 1001, 100)}



        # 读取滑坡和非滑坡数据
        landslide_train = np.load(landslide_train_name)
        nonlandslide_total_train = np.load(nonlandslide_total_train_name)
        landslide_train = remove_features(landslide_train, removed_features_index)
        nonlandslide_total_train = remove_features(nonlandslide_total_train, removed_features_index)

        nonlandslide_train_value = np.random.choice(range(len(nonlandslide_total_train)), int(len(landslide_train)*ratio), replace=False)
        nonlandslide_train = np.array([nonlandslide_total_train[a] for a in nonlandslide_train_value])

        train_data_x = np.concatenate((landslide_train, nonlandslide_train), axis=0)

        train_data_y = np.append(np.ones(len(landslide_train),dtype=int),np.zeros(int(len(landslide_train)*ratio),dtype=int))

        serach = GridSearchCV(estimator=RandomForestClassifier(),param_grid=para_space, scoring='roc_auc', cv=3)
        serach.fit(train_data_x, train_data_y)
        print(serach.best_params_, serach.best_score_)
        # np.savetxt(str(year)+'.txt', serach.best_params_)




