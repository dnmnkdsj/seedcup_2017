import pandas as pd
from pandas import DataFrame
import numpy as np
import csv
import os

'''
    loadMatchData()与loadTeamData()读取所有数据，并进行初步处理
    loadDataSet()根据所采用算法进行数据处理，转化为输入
    
    PS:loadMatchData1()与loadTeamData1()仅使用了csv，感觉不方便处理
'''

base_dir = os.path.abspath(os.path.dirname(__file__))
match_data_URI = os.path.join(base_dir, 'data/matchDataTrain.csv')
team_data_URI = os.path.join(base_dir, 'data/teamData.csv')
test_data_URI = os.path.join(base_dir, 'data/matchDataTest.csv')

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
    team_score[win_team] += k_value(win_team) * (1 - win_probability(win_team, lose_team))
    team_score[lose_team] += k_value(lose_team) * (- win_probability(lose_team, win_team))


def get_team_feature(team, team_data):
    feature = [get_score(team) / 1000]
    for culumn, value in team_data.loc[team].iteritems():
        feature.append(value)
    return feature


def loadDataSet():
    team_data = loadTeamData()
    match_data = loadMatchData()
    feature_data = []
    label = []

    for index, row in match_data.iterrows():
        away_team = row['客场队名']
        away_feature = get_team_feature(away_team, team_data)
        home_team = row['主场队名']
        home_feature = get_team_feature(home_team, team_data)
        home_will_win = row['主场胜负']
        feature_data.append(away_feature + home_feature)
        label.append(home_will_win)
        if home_will_win:
            update_score(home_team, away_team)
        else:
            update_score(away_team, home_team)
    return feature_data, label


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

    return compressTeamData(raw_team_data)


def compressTeamData(team_data):

    print("处理队伍数据...")
    team_data_columns = list(team_data.columns.values)
    print(team_data_columns)
    team_data['上场时间'] *= team_data['出场次数'] / 100
    for col_name in team_data_columns[5:]:
        team_data[col_name] /= team_data['上场时间']

    print(team_data.head())

    compressed_team_data = DataFrame(columns=team_data_columns)
    # 将每个队所有队员信息转化成队伍信息
    for team_name in range(208):  # 共208队
        team_info = team_data[team_data["队名"] == team_name]
        team_info = team_info.apply(lambda x: x.sum())
        team_info['队名'] = team_name
        compressed_team_data = compressed_team_data.append(
            team_info, ignore_index=True)

        # print(team_info.apply(lambda x:x.sum()))

    print(compressed_team_data.head())
    for col_name in team_data_columns[1:5]:
        compressed_team_data.drop(col_name, axis=1, inplace=True)

    compressed_team_data.set_index('队名', drop=True, inplace=True)
    return compressed_team_data


if (__name__ == '__main__'):
    feature_data, label = loadDataSet()
    print("数据构建完成，开始训练")
