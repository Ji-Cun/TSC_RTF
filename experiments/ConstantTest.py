import numpy as np
import math
import datetime
import pandas as pd
from tensorflow import keras
from src.DataIO import loadDataFromTsv
from src.Representation import transform
from src.Segment import getSeriesFeatures
from src.classifiers.CNN import CNN

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import random


def constantTest(dataset='Beef', feature_number=512, random_rate=5):
    x_train_origin, y_train_origin, x_test_origin, y_test_origin = loadDataFromTsv(dataset)

    begin = datetime.datetime.now()
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
    # print('candidate number', len(features))
    # random selection
    randomNumber = feature_number * random_rate
    randomNumber = min(randomNumber, len(features))
    # print('random candidate number', randomNumber)
    randomIndex = random.sample(range(len(features)), randomNumber)
    randomIndex.sort()
    randomFeatures = []
    for index in randomIndex:
        randomFeatures.append(features[index])

    # 基于随机元素变换
    x_train_trans = transform(randomFeatures, x_train_origin)

    if c > 1:
        classifier = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        classifier.fit(x_train_trans, y_train_origin)
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
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
        # print('feature number', len(featuresSelected))
    else:
        x_train_selected = x_train_trans.copy()
        featuresSelected = randomFeatures.copy()

    end = datetime.datetime.now()
    featureSelectionTime = (end - begin).total_seconds()

    x_test_selected = transform(featuresSelected, x_test_origin)
    x_train_mean = x_train_selected.mean()
    x_train_std = x_train_selected.std()
    x_train = (x_train_selected - x_train_mean) / (x_train_std)
    x_test = (x_test_selected - x_train_mean) / (x_train_std)
    # add a dimension to make it multivariate with one dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    nb_classes = len(np.unique(y_train_origin))
    y_train = (y_train_origin - y_train_origin.min()) / (y_train_origin.max() - y_train_origin.min()) * (nb_classes - 1)
    y_test = (y_test_origin - y_test_origin.min()) / (y_test_origin.max() - y_test_origin.min()) * (nb_classes - 1)
    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)

    begin = datetime.datetime.now()
    cnn = CNN(x_train.shape[1:], nb_classes)
    acc_cnn = cnn.fit(x_train, x_test, Y_train, Y_test)
    end = datetime.datetime.now()
    trainTime = (end - begin).total_seconds()
    keras.backend.clear_session()

    return acc_cnn, featureSelectionTime, trainTime


if __name__ == '__main__':

    Datasets = ['Adiac', 'FaceAll', 'Symbols']
    Datasets = ['Adiac', 'FaceAll', 'Symbols']

    df = pd.DataFrame(columns=['dataset', 'itr', 'c', 'acc_cnn', 'featureSelectionTime', 'trainTime'], dtype=object)
    featureNumber = 512
    for itr in range(5):
        for dataset in Datasets:
            for c in [1, 3, 7, 9]:
                print('========================================================')
                print('dataset', dataset)
                print('itr', itr)
                print('c', c)
                print('time', datetime.datetime.now())
                acc_cnn, featureSelectionTime, trainTime = constantTest(dataset=dataset, feature_number=featureNumber,
                                                                        random_rate=c)
                df = df.append({'dataset': dataset, 'itr': itr, 'c': c, 'acc_cnn': acc_cnn,
                                'featureSelectionTime': featureSelectionTime, 'trainTime': trainTime},
                               ignore_index=True)
                resultFileName = "..\\result\\ConstantTestTemp.csv"
                df.to_csv(resultFileName)
    resultFileName = "..\\result\\ConstantTest.csv"
    df.to_csv(resultFileName)
