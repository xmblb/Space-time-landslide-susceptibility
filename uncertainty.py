import numpy as np
import gdal
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
    #13709 7326
    # data2004, geotransform, proj = read_tif('Mask.tif')
    # total_data = np.zeros((13709, 7326, 12))
    # for year in range(2004, 2016):
    #     file_name = 'H:\Taiwan_dynamic_data\data1\RF' + str(year) + '.tif'
    #     data, a, b= read_tif(file_name)
    #     total_data[:,:,year-2004] = data
    #
    # # mean_data = total_data.mean(axis=2)
    # std2 = total_data.std(axis=2)*2
    #
    # # mean_data[data2004[:,:] == -1] = -1
    # std2[data2004[:, :] == -1] = -1
    # # TIF_functions.save_tif(mean_data, "mean_data.tif", geotransform, proj)
    # TIF_functions.save_tif(std2, "std2.tif", geotransform, proj)

    mask, geotransform, proj = read_tif('mask.tif')
    mask = np.reshape(mask, (13709 * 7326,))

    data = np.zeros((47772469, 2))
    mean_data, geotransform, proj = read_tif('H:\Taiwan_dynamic_data\data1\mean_sus.tif')
    mean_data = np.reshape(mean_data, (13709*7326,1))
    std2, geotransform, proj = read_tif('std2.tif')
    std2 = np.reshape(std2, (13709 * 7326,1))

    mean_data_new = mean_data[:,0][mask == 0]
    std2_new = std2[:,0][mask == 0]
    print(mean_data_new.max(), mean_data_new.min())
    print(std2_new.max(), std2_new.min())
    print(mean_data_new.shape)

    #sort and get the rank index
    index = np.argsort(mean_data_new)
    mean_sort = mean_data_new[index]
    std2_sort = std2_new[index]

    data[:,0] = mean_sort
    data[:,1] = std2_sort
    sample_index = np.linspace(0, 47772400, 477725, endpoint=True, dtype=int)

    final_data = data[sample_index, :]
    print(final_data.shape)
    print(final_data.max(), final_data.min())

    # uncertainy_var = np.zeros((477725, 4))
    # rank = np.array(range(1,477726))
    # uncertainy_var[:,0] = final_data[:,0]
    # uncertainy_var[:,1] = final_data[:,0] - final_data[:,1]
    # uncertainy_var[:, 2] = final_data[:,0] + final_data[:,1]
    # uncertainy_var[:, 3] = rank

    # np.savetxt('unvertainty.txt', final_data)
    np.savetxt('uncertainy_var.txt', final_data)



