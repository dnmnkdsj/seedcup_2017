import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import csv
import os
from sklearn import preprocessing

'''
    loadMatchData()与loadTeamData()读取所有数据，并进行初步处理
    loadDataSet()根据所采用算法进行数据处理，转化为输入
    
    PS:loadMatchData1()与loadTeamData1()仅使用了csv，感觉不方便处理
'''

base_dir=os.path.abspath(os.path.dirname(__file__))
match_data_URI=os.path.join(base_dir,'data/matchDataTrain.csv')
team_data_URI=os.path.join(base_dir,'data/teamData.csv')
test_data_URI = os.path.join(base_dir, 'data/matchDataTest.csv')
output_URI = os.path.join(base_dir, 'data/predictPro.csv')

base_score = 1600
team_score = {}
global_var={}



def loadDataSet(team_features=['投篮命中率','投篮命中次数',
                               '投篮出手次数','篮板总数','助攻',
                               '抢断','盖帽','失误','犯规','得分']):
    '''
    可定制所要特征(仅限于team表中的特征

    :param team_features:
    所需要的team表中的特征列表
    :param match_features:
    match表中的特征默认为主、客场胜负场数，如需要更改对返回只调用drop()
    :return:
    dataSet为DataFrame类型，存储特征，为主场队信息减去客场队信息（其他处理另行定制
    labelSet为Series类型，存储胜负
    '''
    raw_team_data=loadTeamData()
    raw_match_data,z,k,n=loadMatchData()
    handled_team_data=compressTeamData(raw_team_data)
    handled_team_data['作主场胜率']=Series(z)
    handled_team_data['作客场胜率']=Series(k)
    handled_team_data['比赛场数']=Series(n)
    '''
    for col_name in [ '投篮命中次数',
                     '投篮出手次数', '三分命中次数','三分出手次数',
                     '罚球命中次数','罚球出手次数','篮板总数','前场篮板','后场篮板', '助攻',
                     '抢断', '盖帽', '失误', '犯规', '得分']:
        handled_team_data[col_name] /= (handled_team_data['比赛场数'])
    '''
    handled_team_data.fillna(0)

    match_features = ['客场前胜场数', '客场前负场数',
                      '主场前胜场数', '主场前负场数','主对客历史胜率']


    dataSet_rows=[]
    labelSet=[]

    for index,row in raw_match_data.iterrows():
        home_will_win = row['主场胜负']
        if home_will_win:
            update_score(row['主场队名'], row['客场队名'])
        else:
            update_score(row['客场队名'], row['主场队名'])

    score=[0 for i in range(208)]
    for index,value in team_score.items():
        score[index]=value

    handled_team_data['等级分']=Series(score)

    rank_dict = {}
    score.sort(reverse=True)
    k = 1
    for i in score:
        rank_dict[i] = k
        k += 1
    seed = [rank_dict[row['等级分']] for index, row in handled_team_data.iterrows()]
    handled_team_data['seed'] = Series(seed)
    print(handled_team_data.loc[0])

    team_data_columns = list(handled_team_data.columns.values)
    print(team_data_columns)
    for col_name in team_data_columns:
        if col_name not in team_features:
            handled_team_data.drop(col_name,axis=1,inplace=True)
    print(handled_team_data.head())





    for index,row in raw_match_data.iterrows():
        team_data_temp_z=handled_team_data.loc[row['主场队名']].copy()
        team_data_temp_k = handled_team_data.loc[row['客场队名']].copy()
        #               -handled_team_data.loc[row['客场队名']]
        #team_data_temp1=handled_team_data.loc[row['主场队名']]\
        #               -handled_team_data.loc[row['客场队名']]
        try:
            team_data_temp_z.drop('作客场胜率',inplace=True)
            team_data_temp_k.drop('作主场胜率',inplace=True)
        except:
            pass
        try:
            team_data_temp_z['等级分']+=100
        except:
            pass

        match_data_temp=row.loc['主场前胜场数':'主场前负场数']
        win_prob=win_probability(row['主场队名'],row['客场队名'])

        features=[]
        features.append(win_prob)
        if '前胜率' in team_features:
            if row['客场前负场数']+row['客场前胜场数']!=0:
                features.append(row['客场前胜场数']/(row['客场前负场数']+row['客场前胜场数']))
            else:
                features.append(0.5)
            if row['主场前负场数']+row['主场前胜场数']:
                features.append(row['主场前胜场数']/(row['主场前负场数'] + row['主场前胜场数']))
            else:
                features.append(0.5)
        features.extend(list(match_data_temp))
        #features.extend(list(team_data_temp_z/team_data_temp_k))
        features.extend(list(team_data_temp_z)+list(team_data_temp_k))
        dataSet_rows.append(features)
        labelSet.append(row['主场胜负'])

    dataSet=DataFrame(dataSet_rows)
    #preprocessing.scale(dataSet, copy=False)

    labelSet=Series(labelSet)
    testSet=[]

    raw_test_data=loadTestData(raw_match_data)
    print(raw_match_data.head())
    print(raw_test_data.head())
    testSet_rows=[]
    for index,row in raw_test_data.iterrows():
        team_data_temp_z = handled_team_data.loc[row['主场队名']].copy()
        team_data_temp_k = handled_team_data.loc[row['客场队名']].copy()
        #               -handled_team_data.loc[row['客场队名']]
        # team_data_temp1=handled_team_data.loc[row['主场队名']]\
        #               -handled_team_data.loc[row['客场队名']]
        try:
            team_data_temp_z.drop('作客场胜率', inplace=True)
            team_data_temp_k.drop('作主场胜率', inplace=True)
        except:
            pass
        try:
            team_data_temp_z['等级分'] += 100
        except:
            pass

        match_data_temp = row.loc['主场前胜场数':'主场前负场数']
        win_prob = win_probability(row['主场队名'], row['客场队名'])

        features = []
        features.append(win_prob)
        if '前胜率' in team_features:
            if row['客场前负场数'] + row['客场前胜场数'] != 0:
                features.append(row['客场前胜场数'] / (row['客场前负场数'] + row['客场前胜场数']))
            else:
                features.append(0.5)
            if row['主场前负场数'] + row['主场前胜场数']:
                features.append(row['主场前胜场数'] / (row['主场前负场数'] + row['主场前胜场数']))
            else:
                features.append(0.5)
        features.extend(list(match_data_temp))
        # features.extend(list(team_data_temp_z/team_data_temp_k))
        features.extend(list(team_data_temp_z) + list(team_data_temp_k))
        testSet_rows.append(features)


    testSet=DataFrame(testSet_rows)

    print('the dataSet and labelSet are:')
    print(dataSet.head())
    print(labelSet.head())
    print(list(dataSet.columns.values))

    dataSet.fillna(0)

    return dataSet,labelSet,testSet

def compressTeamData(raw_team_data):
    team_data_columns = list(raw_team_data.columns.values)
    # print(team_data_columns)

    #raw_team_data=raw_team_data[raw_team_data['上场时间']*raw_team_data['出场次数']>81]

    for col_name in team_data_columns[4:]:
        raw_team_data[col_name] *= raw_team_data['出场次数']

    # print(raw_team_data.head())
    


    handled_team_data = DataFrame(columns=team_data_columns)
    # 将每个队所有队员信息转化成队伍信息
    for team_name in range(208):  # 共208队
        team_info = raw_team_data.loc[raw_team_data["队名"] == team_name ]

        handled_team_data = handled_team_data.append(
            team_info.apply(lambda x: x.sum()), ignore_index=True)
        handled_team_data['人数']=len(team_info)
        get_score(team_name)

        # print(team_info.apply(lambda x:x.sum()))

    for col_name in team_data_columns[6:]:
        handled_team_data[col_name] /= 82
    handled_team_data['百回合得分']=handled_team_data['得分']/handled_team_data['回合数']*100
    handled_team_data['百回合失误']=handled_team_data['失误']/handled_team_data['回合数']*100
    handled_team_data['DWS']=1/handled_team_data['百回合得分']
    handled_team_data['投篮命中率'] = handled_team_data['投篮命中次数'] / handled_team_data['投篮出手次数']
    handled_team_data['三分命中率'] = handled_team_data['三分命中次数'] / handled_team_data['三分出手次数']
    handled_team_data['罚球命中率'] = handled_team_data['罚球命中次数'] / handled_team_data['罚球出手次数']
    handled_team_data['人均进攻篮板比例']=handled_team_data['前场篮板']/handled_team_data['篮板总数']/handled_team_data['人数']
    handled_team_data['人均防守篮板比例'] = handled_team_data['后场篮板'] / handled_team_data['篮板总数']/handled_team_data['人数']
    handled_team_data['罚球比例'] = handled_team_data['罚球出手次数'] / handled_team_data['投篮出手次数']
    handled_team_data['三分比例'] = handled_team_data['三分出手次数'] / handled_team_data['投篮出手次数']

    match_data=pd.read_csv(match_data_URI)


    print('compressing data:')
    print(handled_team_data.head(10))

    return handled_team_data


def loadMatchData():
    '''
    :return:
    raw_match_data为pandas内置的DataFrame类型
    列标签为 （import 然后运行可知）
    '''
    raw_match_data=pd.read_csv(match_data_URI)

    cols_to_change=["客场本场前战绩","主场本场前战绩","比分（客场:主场）"]

    print(raw_match_data.head(30))

    #提取胜场数和负场数
    dataframe_temp1=raw_match_data[cols_to_change[0]].\
        str.extract('(\d+)胜(\d+)负',expand=True)
    dataframe_temp1.rename(columns={0:"客场前胜场数",1:"客场前负场数"},
                           inplace=True)

    dataframe_temp2 = raw_match_data[cols_to_change[1]]. \
        str.extract('(\d+)胜(\d+)负',expand=True)
    dataframe_temp2.rename(columns={0: "主场前胜场数", 1: "主场前负场数"},
                           inplace=True)

    #提取比分
    dataframe_temp3=raw_match_data[cols_to_change[2]].\
        str.extract('(\d+):(\d+)',expand=True)
    dataframe_temp3.rename(columns={0: "客场本场得分", 1: "主场本场得分"},
                           inplace=True)
    dataframe_temp3['客场胜负']=dataframe_temp3["客场本场得分"]\
                            >dataframe_temp3['主场本场得分']

    #获取胜负情况
    dataframe_temp3['客场胜负']=dataframe_temp3['客场胜负']\
        .replace({True:1,False:0})
    dataframe_temp3['主场胜负']=dataframe_temp3['客场胜负']
    dataframe_temp3['主场胜负']=dataframe_temp3['主场胜负']\
        .replace({1:0,0:1})

    #将处理后的数据插入raw_match_data中
    for col_name in cols_to_change:
        del raw_match_data[col_name]

    for frame in [dataframe_temp1,dataframe_temp2,dataframe_temp3]:
        for colname in list(frame.columns.values):
            raw_match_data[colname]=frame[colname].astype(int)

    rates_z_to_k=[]
    rates_of_z=[]
    rates_of_k=[]
    num_of_match=[]

    print(raw_match_data.head(30))
    print(raw_match_data['主场胜负'].value_counts())

    for name in range(208):
        data_temp = raw_match_data[raw_match_data['主场队名'] == name]
        num_of_match.append(len(data_temp))
        if data_temp.empty:
            rates_of_z.append(0)
        else:
            rates_of_z.append(int(data_temp['主场胜负'].value_counts()[1]) / len(data_temp))
        data_temp = raw_match_data[raw_match_data['客场队名'] == name]
        num_of_match[name]+=len(data_temp)
        if data_temp.empty:
            rates_of_k.append(1)
        else:
            rates_of_k.append(int(data_temp['客场胜负'].value_counts()[1]) / len(data_temp))

    print(raw_match_data.head())
    return raw_match_data,rates_of_z,rates_of_k,num_of_match

def loadTeamData():
    '''
    :return:
     同loadMatchData（）
    '''
    raw_team_data=pd.read_csv(team_data_URI)

    cols_to_change=['投篮命中率','三分命中率','罚球命中率']

    #将百分数转化为浮点数
    for col_name in cols_to_change:
        str_to_float=raw_team_data[col_name].str.strip('%')\
                         .astype(float)/100
        raw_team_data[col_name]=str_to_float

    raw_team_data.fillna(0,inplace=True)

    raw_team_data['回合数']=raw_team_data['投篮出手次数']+0.4*raw_team_data['罚球出手次数']\
                           -raw_team_data['前场篮板']+raw_team_data['失误']

    return raw_team_data

def loadTestData(match_data):
    raw_test_data=pd.read_csv(test_data_URI)

    cols_to_change = ["客场本场前战绩", "主场本场前战绩"]

    # 提取胜场数和负场数
    dataframe_temp1 = raw_test_data[cols_to_change[0]]. \
        str.extract('(\d+)胜(\d+)负', expand=True)
    dataframe_temp1.rename(columns={0: "客场前胜场数", 1: "客场前负场数"},
                           inplace=True)

    dataframe_temp2 = raw_test_data[cols_to_change[1]]. \
        str.extract('(\d+)胜(\d+)负', expand=True)
    dataframe_temp2.rename(columns={0: "主场前胜场数", 1: "主场前负场数"},
                           inplace=True)

    # 将处理后的数据插入raw_match_data中
    for col_name in cols_to_change:
        del raw_test_data[col_name]

    for frame in [dataframe_temp1, dataframe_temp2]:
        for colname in list(frame.columns.values):
            raw_test_data[colname] = frame[colname].astype(int)

    return raw_test_data

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

def write_pred_result(result):
    with open(output_URI, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['主场赢得比赛的置信度'])
        writer.writerows(result)


