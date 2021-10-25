import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(filePath):
    '''
    read CSV file
    :param filePath:
    :return: x, y
    '''
    data = pd.read_csv(filePath)
    data = data.values
    data_x = data[:,0]
    data_y = data[:,1]
    return data_x, data_y

if __name__ == '__main__':

    data_x, data_y = read_data('lowess.csv')

    low = sm.nonparametric.lowess
    results = low(data_y, data_x)
    print(results.shape, results)
    asd = pd.DataFrame(data=results)
    asd.to_csv('res.csv', index=False, header=False)
    # plot curve
    plt.scatter(data_x,data_y)
    #plot scatter
    plt.plot(results[:,0], results[:,1])
    plt.show()

