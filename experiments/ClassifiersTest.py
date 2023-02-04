import numpy as np
import math
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from src.DataIO import loadDataFromTsv
from src.Representation import transform
from src.Segment import getSeriesFeatures
from src.classifiers.FCN import FCN
from src.classifiers.CNN import CNN
from src.classifiers.VGG16 import VGG

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import random


"""
dataset: 测试的数据集
feature_number: 最终选择特征的数目
random_rate：随机选择特征的比率
segAvgLen：期望的平均Segment长度，默认为10
"""

def accuracyExperiemnt(dataset='Beef', feature_number=512,random_rate=5):

    x_train_origin, y_train_origin, x_test_origin, y_test_origin = loadDataFromTsv(dataset)

    # generate intuitive temporal features
    features = []

    # 最大长度为100或者时间序列长度、最小长度设为5，期望的分段数为长度除以10和5之间的较大值
    maxLength = min(len(x_train_origin[0]), 100)
    minLength = 5
    segNumber = max(math.ceil(len(x_train_origin[0]) / 10), 5)
    for seriesId in range(len(x_train_origin)):
        values = x_train_origin[seriesId]
        seriesFeatures = getSeriesFeatures(seriesId, values, segNumber, maxLength, minLength)
        for feature in seriesFeatures:
            features.append(feature)
    print('candidate number', len(features))

    # random selection
    randomNumber=feature_number*random_rate
    randomNumber=min(randomNumber,len(features))
    print('random candidate number', randomNumber)
    randomIndex=random.sample(range(len(features)),randomNumber)
    randomIndex.sort()

    randomFeatures=[]
    for index in randomIndex:
        randomFeatures.append(features[index])

    # 基于随机元素变换
    x_train_trans = transform(randomFeatures, x_train_origin)

    classifier = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    classifier.fit(x_train_trans, y_train_origin)

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]


    nb_classes = len(np.unique(y_test_origin))

    y_train = (y_train_origin - y_train_origin.min()) / (y_train_origin.max() - y_train_origin.min()) * (nb_classes - 1)
    y_test = (y_test_origin - y_test_origin.min()) / (y_test_origin.max() - y_test_origin.min()) * (nb_classes - 1)
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)

    threshold = 0
    if (len(importances) > feature_number):
        threshold = importances[indices[feature_number - 1]]
    if threshold == 0:
        threshold = min(importance for importance in importances if importance > 0)

    x_train_selected = x_train_trans[:, importances >= threshold]
    selectedIndex = np.argwhere(importances >= threshold)
    featuresSelected = []
    for index in selectedIndex:
        feature = randomFeatures[index[0]]
        featuresSelected.append(feature)
    print('feature number', len(featuresSelected))

    x_test_selected = transform(featuresSelected, x_test_origin)

    x_train_mean = x_train_selected.mean()
    x_train_std = x_train_selected.std()
    x_train = (x_train_selected - x_train_mean) / (x_train_std)
    x_test = (x_test_selected - x_train_mean) / (x_train_std)

    # add a dimension to make it multivariate with one dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    cnn = CNN(x_train.shape[1:], nb_classes)
    cnn.fit(x_train, x_test, Y_train, Y_test)
    y_pred_cnn = cnn.predict(x_test)
    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    print("acc_cnn:", acc_cnn)

    '''
    fcn = FCN(x_train.shape[1:], nb_classes)
    fcn.fit(x_train, x_test, Y_train, Y_test)
    y_pred_fcn = fcn.predict(x_test)
    acc_fcn = accuracy_score(y_test, y_pred_fcn)
    print("acc_fcn:", acc_fcn)
    

    vgg = VGG(x_train.shape[1:], nb_classes)
    vgg.fit(x_train, x_test, Y_train, Y_test)
    y_pred_vgg = vgg.predict(x_test)
    acc_vgg = accuracy_score(y_test, y_pred_vgg)
    print("acc_vgg:", acc_vgg)
    '''

    keras.backend.clear_session()

    return acc_cnn, acc_vgg


if __name__ == '__main__':

    UCR_43 = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'CricketX', 'CricketY',
              'CricketZ', 'DiatomSizeReduction', 'ECG200', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR',
              'FiftyWords', 'Fish', 'GunPoint', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lightning2',
              'Lightning7', 'Mallat', 'MedicalImages', 'MoteStrain', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface1',
              'SonyAIBORobotSurface2', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'Trace',
              'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
              'Wafer', 'WordSynonyms', 'Yoga']

    UCR_43 = ['Adiac', 'Beef', 'ChlorineConcentration', 'Coffee', 'DiatomSizeReduction',
              'ECGFiveDays', 'FaceAll',  'GunPoint',  'ItalyPowerDemand',
              'Lightning7', 'MedicalImages', 'MoteStrain',  'SonyAIBORobotSurface1',
              'Symbols',  'Trace', 'TwoLeadECG']

    UCR_43 = ['Adiac', 'ECGFiveDays', 'FaceAll',  'Symbols',  'Trace', 'TwoLeadECG']

    UCR_43 = ['ECGFiveDays', 'FaceAll', 'Symbols', 'Trace', 'TwoLeadECG']

    df = pd.DataFrame(columns=['dataset', 'itr','feature number', 'acc_cnn'], dtype=object)
    for dataset in UCR_43:
        for itr in range(2):
          for featureNumber in [64, 128, 256, 512, 1024]:
            print('========================================================')
            print('dataset', dataset)
            print('itr', itr)
            print('feature number', featureNumber)
            print('time', datetime.datetime.now())
            acc_cnn = accuracyExperiemnt(dataset=dataset,feature_number=featureNumber)
            df = df.append({'dataset': dataset, 'itr': itr,'feature number': featureNumber, 'acc_cnn': acc_cnn}, ignore_index=True)
            resultFileName = "..\\result\\FeatureNumComparisionTemp.csv"
            df.to_csv(resultFileName)
    resultFileName = "..\\result\\FeatureNumComparision.csv"
    df.to_csv(resultFileName)