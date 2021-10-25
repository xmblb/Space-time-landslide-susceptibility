import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import csv

def remove_features(original_data, index):
    '''remove redundant features'''
    output = np.zeros((len(original_data), len(index)))
    for i in range(len(index)):
        output[:,i] = original_data[:,index[i]]
    return output

def random_sample(input_data, number):
    '''randomly select samples'''
    data_index = np.random.choice(range(len(input_data)), number, replace=False)
    sample_data = np.array([input_data[a,:] for a in data_index])
    return sample_data

def get_train_test(landslide_train, nonlandslide_total_train, ratio):
    '''generate traning and test sets for sensitivity analysis'''
    nonlandslide_new_total_train, nonlandslide_new_total_test = train_test_split(nonlandslide_total_train,
                                                                                 test_size=0.2)

    number_landslide_train = int(len(landslide_train)*0.1)

    landslide_sub = random_sample(landslide_train, number_landslide_train*2)
    landslide_new_train, landslide_new_test = train_test_split(landslide_sub,test_size=0.5)

    nonlandslide_new_train = random_sample(nonlandslide_new_total_train, int(number_landslide_train * ratio))
    nonlandslide_new_test = random_sample(nonlandslide_new_total_test, number_landslide_train)

    # print(landslide_new_train.shape, nonlandslide_new_train.shape)
    train_x = np.concatenate((landslide_new_train, nonlandslide_new_train), axis=0)
    test_x = np.concatenate((landslide_new_test, nonlandslide_new_test), axis=0)

    train_y = np.append(np.ones(len(landslide_new_train), dtype=int), np.zeros(len(nonlandslide_new_train), dtype=int))
    test_y = np.append(np.ones(len(landslide_new_test), dtype=int), np.zeros(len(nonlandslide_new_test), dtype=int))

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    year = 2008

    landslide_train = np.load('train_landslide'+str(year) + '.npy')
    nonlandslide_total_train = np.load('train_nonlandslide'+str(year) + '.npy')
    removed_features_index = np.load('retained_features'+ str(year) + '.npy')


    landslide_train = remove_features(landslide_train, removed_features_index)
    nonlandslide_total_train = remove_features(nonlandslide_total_train, removed_features_index)

    producer_acc_list = []
    user_acc_list = []


    for ratio in range(10, 51):
        ratio = ratio/10.0
        print(ratio)
        produce_temp = []
        use_temp = []
        for i in range(10):
            # np.random.seed(i)
            train_x, train_y, test_x, test_y = get_train_test(landslide_train, nonlandslide_total_train, ratio)
            model = RandomForestClassifier(n_estimators=500)
            model.fit(train_x, train_y)

            pred_class = model.predict(test_x)
            # print(pred_class)
            tn, fp, fn, tp = metrics.confusion_matrix(test_y, pred_class).ravel()
            # print(metrics.accuracy_score(test_y, pred_class))
            producer_acc = metrics.recall_score(test_y,pred_class)
            user_acc = metrics.precision_score(test_y,pred_class)
            produce_temp.append(producer_acc)
            use_temp.append(user_acc)
        print(np.average(produce_temp), np.average(use_temp))
        producer_acc_list.append(produce_temp)
        user_acc_list.append(use_temp)
    producer_acc_list = pd.DataFrame(producer_acc_list)
    user_acc_list = pd.DataFrame(user_acc_list)
    # print(producer_acc_list)
    # print(user_acc_list)
    producer_acc_list.to_csv('producer_acc'+str(year)+'.csv', index=False, header=False)
    user_acc_list.to_csv('user_acc'+str(year)+'.csv', index=False, header=False)

