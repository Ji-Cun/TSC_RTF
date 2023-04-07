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


def candicateNumberTest(dataset='Beef', feature_number=512, random_rate=5):
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

    # print('candidate number', len(features))
    candicateNumber = len(features)


    # random selection
    randomNumber = feature_number * random_rate
    randomNumber = min(randomNumber, len(features))
    # print('random candidate number', randomNumber)

    randomIndex = random.sample(range(len(features)), randomNumber)
    randomIndex.sort()
    randomFeatures = []
    for index in randomIndex:
        randomFeatures.append(features[index])

    return candicateNumber, randomNumber


if __name__ == '__main__':
    UCR_43 = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'CricketX', 'CricketY',
              'CricketZ', 'DiatomSizeReduction', 'ECG200', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR',
              'FiftyWords', 'Fish', 'GunPoint', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lightning2',
              'Lightning7', 'Mallat', 'MedicalImages', 'MoteStrain', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface1',
              'SonyAIBORobotSurface2', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'Trace',
              'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
              'Wafer', 'WordSynonyms', 'Yoga']

    dataset_feature = {'Adiac': 2048, 'Beef': 1024, 'CBF': 2048, 'ChlorineConcentration': 64, 'CinCECGTorso': 1024,
                       'Coffee': 1024, 'CricketX': 1024, 'CricketY': 1024, 'CricketZ': 2048, 'DiatomSizeReduction': 512,
                       'ECG200': 1024, 'ECGFiveDays': 128, 'FaceAll': 512, 'FaceFour': 512, 'FacesUCR': 1024,
                       'FiftyWords': 2048, 'Fish': 1024, 'GunPoint': 2048, 'Haptics': 1024, 'InlineSkate': 2048,
                       'ItalyPowerDemand': 128, 'Lightning2': 2048, 'Lightning7': 1024, 'Mallat': 1024,
                       'MedicalImages': 512, 'MoteStrain': 512, 'OliveOil': 512, 'OSULeaf': 2048,
                       'SonyAIBORobotSurface1': 64, 'SonyAIBORobotSurface2': 256, 'StarLightCurves': 1024,
                       'SwedishLeaf': 2048, 'Symbols': 512, 'SyntheticControl': 1024, 'Trace': 64, 'TwoLeadECG': 256,
                       'TwoPatterns': 2048, 'UWaveGestureLibraryX': 2048, 'UWaveGestureLibraryY': 2048,
                       'UWaveGestureLibraryZ': 2048, 'Wafer': 256, 'WordSynonyms': 2048, 'Yoga': 1024}
    df = pd.DataFrame(columns=['dataset', 'feature number', 'candicateNumber','randomNumber'], dtype=object)
    for dataset in UCR_43:
        featureNumber = dataset_feature[dataset]
        print('========================================================')
        print('dataset', dataset)
        print('feature number', featureNumber)
        print('time', datetime.datetime.now())
        candicateNumber, randomNumber = candicateNumberTest(dataset=dataset, feature_number=featureNumber)
        df = df.append({'dataset': dataset, 'feature number': featureNumber, 'candicateNumber': candicateNumber,'randomNumber': randomNumber},
                               ignore_index=True)
    resultFileName = "..\\result\\CondicateNumberTest.csv"
    df.to_csv(resultFileName)

