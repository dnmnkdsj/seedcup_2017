import pandas as pd
from pandas import DataFrame
import numpy as np
import csv
import os
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr

'''
    loadMatchData()与loadTeamData()读取所有数据，并进行初步处理
    loadDataSet()根据所采用算法进行数据处理，转化为输入
    
    PS:loadMatchData1()与loadTeamData1()仅使用了csv，感觉不方便处理
'''

base_dir = os.path.abspath(os.path.dirname(__file__))
match_data_URI = os.path.join(base_dir, 'data/matchDataTrain.csv')
team_data_URI = os.path.join(base_dir, 'data/teamData.csv')
test_data_URI = os.path.join(base_dir, 'data/matchDataTest.csv')
output_URI = os.path.join(base_dir, 'data/predictPro.csv')

base_score = 1600
team_score = {}


def get_score(team):
    if (team not in team_score):
        team_score[team] = base_score
    return team_score[team]


def win_probability(team_a, team_b):
    score_diff = get_score(team_b) - get_score(team_a)
    exp = score_diff / 400
    return 1 / (1 + 10 ** exp)


def k_value(team):
    score = get_score(team)
    if score < 2100:
        win_k = 32
    elif score < 2400:
        win_k = 24
    else:
        win_k = 16
    return win_k


def update_score(win_team, lose_team):
    win_prob = win_probability(win_team, lose_team)
    team_score[win_team] += round(k_value(win_team) * (1 - win_probability(win_team, lose_team)))
    team_score[lose_team] += round(k_value(lose_team) * (- win_probability(lose_team, win_team)))


def get_team_feature(team, team_data):
    feature = [get_score(team) ]
    for culumn, value in team_data.loc[team].iteritems():
        feature.append(value)

    return feature

def connect_data(match, team_data):
    away_feature = get_team_feature(match['客场队名'], team_data)
    away_feature.append(match['客场胜负比'])
    home_feature = get_team_feature(match['主场队名'], team_data)
    home_feature.append(match['主场胜负比'])
    return away_feature + home_feature

def loadDataSet():
    team_data = loadTeamData()
    match_data = loadMatchData()
    feature_data = []
    label = []

    for index, row in match_data.iterrows():
        home_will_win = row['主场胜负']
        feature_data.append(connect_data(row, team_data))
        label.append(home_will_win)
        if home_will_win:
            update_score(row['主场队名'], row['客场队名'])
        else:
            update_score(row['客场队名'], row['主场队名'])
    print("team_data")
    print(team_data.head())
    return team_data, feature_data, label


def loadMatchData():
    '''
    :return:
    raw_match_data为pandas内置的DataFrame类型
    列标签为 （import 然后运行可知）
    '''
    raw_match_data = pd.read_csv(match_data_URI)

    print("加载比赛记录...")

    cols_to_change = ["客场本场前战绩", "主场本场前战绩", "比分（客场:主场）"]

    # 提取胜场数和负场数
    dataframe_temp1 = raw_match_data[cols_to_change[0]]. \
        str.extract('(\d+)胜(\d+)负', expand=True)
    dataframe_temp1.rename(columns={0: "客场前胜场数", 1: "客场前负场数"},
                           inplace=True)

    dataframe_temp2 = raw_match_data[cols_to_change[1]]. \
        str.extract('(\d+)胜(\d+)负', expand=True)
    dataframe_temp2.rename(columns={0: "主场前胜场数", 1: "主场前负场数"},
                           inplace=True)

    # 提取比分
    dataframe_temp3 = raw_match_data[cols_to_change[2]]. \
        str.extract('(\d+):(\d+)', expand=True)
    dataframe_temp3.rename(columns={0: "客场本场得分", 1: "主场本场得分"},
                           inplace=True)
    dataframe_temp3['主场胜负'] = dataframe_temp3["客场本场得分"] \
                              < dataframe_temp3['主场本场得分']

    # 获取胜负情况
    dataframe_temp3['主场胜负'] = dataframe_temp3['主场胜负'] \
        .replace({True: 1, False: 0})

    # 将处理后的数据插入raw_match_data中
    for col_name in cols_to_change:
        del raw_match_data[col_name]

    for frame in [dataframe_temp1, dataframe_temp2, dataframe_temp3]:
        for colname in list(frame.columns.values):
            raw_match_data[colname] = frame[colname]

    raw_match_data['主场胜负比'] = raw_match_data['主场前胜场数'].astype(int) / (raw_match_data['主场前负场数'].astype(int) + 1)
    raw_match_data['客场胜负比'] = raw_match_data['客场前负场数'].astype(int) / (raw_match_data['客场前负场数'].astype(int) + 1)
    raw_match_data.loc[:, ['主场胜负比', '客场胜负比']] = preprocessing.scale(raw_match_data.loc[:, ['主场胜负比', '客场胜负比']])

    print(raw_match_data.loc[:, ['主场胜负比', '客场胜负比']])
    raw_match_data.fillna(0, inplace=True)
    return raw_match_data


def loadTeamData():
    '''
    :return:
    同loadMatchData（）
    '''
    raw_team_data = pd.read_csv(team_data_URI)

    print("加载队伍数据...")
    cols_to_change = ['投篮命中率', '三分命中率', '罚球命中率']

    # 将百分数转化为浮点数
    for col_name in cols_to_change:
        str_to_float = raw_team_data[col_name].str.strip('%') \
                           .astype(float) / 100
        raw_team_data[col_name] = str_to_float

    raw_team_data.fillna(0, inplace=True)

    print("raw_team_data:")
    print(np.sum(raw_team_data))
    return compressTeamData(raw_team_data)


def compressTeamData(team_data):
    print("压缩队伍数据...")
    team_data_columns = list(team_data.columns.values)
    for col_name in team_data_columns[4:]:
        team_data[col_name] *= team_data['出场次数']

    compressed_team_data = DataFrame(columns=team_data_columns)
    # 将每个队所有队员信息转化成队伍信息
    for team_name in range(208):  # 共208队
        team_info = team_data[team_data["队名"] == team_name]
        team_info = team_info.apply(lambda x: x.sum())
        team_info['队名'] = team_name
        compressed_team_data = compressed_team_data.append(
            team_info, ignore_index=True)

        # print(team_info.apply(lambda x:x.sum()))

    for col_name in team_data_columns[6:]:
        compressed_team_data[col_name] /= (compressed_team_data['上场时间'])

    compressed_team_data['投篮命中率'] = compressed_team_data['投篮命中次数'] / compressed_team_data['投篮出手次数']
    compressed_team_data['三分命中率'] = compressed_team_data['三分命中次数'] / compressed_team_data['三分出手次数']
    compressed_team_data['罚球命中率'] = compressed_team_data['罚球命中次数'] / compressed_team_data['罚球出手次数']

    print("compressed_team_data:")
    print(compressed_team_data.head())
    compressed_team_data.fillna(0, inplace=True)
    for col_name in team_data_columns[0:5]:
        compressed_team_data.drop(col_name, axis=1, inplace=True)


    #preprocessing.scale(compressed_team_data, copy=False)

    print("compressed_team_data1:")
    print(compressed_team_data.head())
    return compressed_team_data


def load_test_feature(team_data):
    test_data = load_test_data()
    feature_data = []

    for index, row in test_data.iterrows():
        feature_data.append(connect_data(row, team_data))
    return feature_data


def load_test_data():
    raw_match_data = pd.read_csv(test_data_URI)

    print("加载测试数据...")

    cols_to_change = ["客场本场前战绩", "主场本场前战绩"]

    # 提取胜场数和负场数
    dataframe_temp1 = raw_match_data[cols_to_change[0]]. \
        str.extract('(\d+)胜(\d+)负', expand=True)
    dataframe_temp1.rename(columns={0: "客场前胜场数", 1: "客场前负场数"},
                           inplace=True)

    dataframe_temp2 = raw_match_data[cols_to_change[1]]. \
        str.extract('(\d+)胜(\d+)负', expand=True)
    dataframe_temp2.rename(columns={0: "主场前胜场数", 1: "主场前负场数"},
                           inplace=True)

    # 将处理后的数据插入raw_match_data中
    for col_name in cols_to_change:
        del raw_match_data[col_name]

    for frame in [dataframe_temp1, dataframe_temp2]:
        for colname in list(frame.columns.values):
            raw_match_data[colname] = frame[colname]

    raw_match_data['主场胜负比'] = raw_match_data['主场前胜场数'].astype(int) / (raw_match_data['主场前负场数'].astype(int) + 1)
    raw_match_data['客场胜负比'] = raw_match_data['客场前胜场数'].astype(int) / (raw_match_data['客场前负场数'].astype(int) + 1)
    raw_match_data.loc[:, ['主场胜负比', '客场胜负比']] = preprocessing.scale(raw_match_data.loc[:, ['主场胜负比', '客场胜负比']])
    return raw_match_data


def write_pred_result(result):
    with open(output_URI, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['主场赢得比赛的置信度'])
        writer.writerows(result)


if (__name__ == '__main__'):
    team_data, feature_data, label = loadDataSet()
    print("数据构建完成,开始训练")
    test_feature = load_test_feature(team_data)

    # classifier = svm.SVC(kernel='rbf', probability=True, gamma='auto')
    # classifier.fit(feature_data, label)
    # result = classifier.predict_proba(test_feature)


    l = lr()
    l.fit(feature_data, label)
    result = l.predict_proba(test_feature)

    single_result = [[x[0]] for x in result]

    write_pred_result(single_result)
