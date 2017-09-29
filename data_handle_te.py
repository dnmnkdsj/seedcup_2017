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
    raw_match_data,z,k=loadMatchData()
    team_data_columns = list(raw_team_data.columns.values)
    handled_team_data=compressTeamData(raw_team_data)
    handled_team_data['作主场胜率']=Series(z)
    handled_team_data['作客场胜率']=Series(k)
    handled_team_data.fillna(0)

    match_features = ['客场前胜场数', '客场前负场数',
                      '主场前胜场数', '主场前负场数','主对客历史胜率']

    for col_name in team_data_columns:
        if col_name not in team_features:
            handled_team_data.drop(col_name,axis=1,inplace=True)

    dataSet_rows=[]
    labelSet=[]

    for index,row in raw_match_data.iterrows():
        team_data_temp=handled_team_data.loc[row['主场队名']]
        #               -handled_team_data.loc[row['客场队名']]
        match_data_temp=row.loc['客场前胜场数':'主对客历史胜率']

        dataSet_rows.append(list(team_data_temp)
                            +list(handled_team_data.loc[row['客场队名']])
                            +list(match_data_temp))
        labelSet.append(row['主场胜负'])
        '''
        match_data_temp_list = list(match_data_temp)
        match_data_temp_list[0],match_data_temp_list[2]=\
            match_data_temp_list[2],match_data_temp_list[0]
        match_data_temp_list[1], match_data_temp_list[3] = \
            match_data_temp_list[3], match_data_temp_list[1]
        dataSet_rows.append(list(-team_data_temp))
        labelSet.append(row['客场胜负'])
        '''

    dataSet=DataFrame(dataSet_rows)
    #preprocessing.scale(dataSet, copy=False)

    labelSet=Series(labelSet)
    print('the dataSet and labelSet are:')
    print(dataSet.head())
    print(labelSet.head())
    print(list(dataSet.columns.values))

    return dataSet,labelSet





def compressTeamData(raw_team_data):
    team_data_columns = list(raw_team_data.columns.values)
    # print(team_data_columns)

    for col_name in team_data_columns[4:]:
        raw_team_data[col_name] *= raw_team_data['出场次数']

    # print(raw_team_data.head())

    handled_team_data = DataFrame(columns=team_data_columns)
    # 将每个队所有队员信息转化成队伍信息
    for team_name in range(208):  # 共208队
        team_info = raw_team_data.loc[raw_team_data["队名"] == team_name]
        handled_team_data = handled_team_data.append(
            team_info.apply(lambda x: x.sum()), ignore_index=True)

        # print(team_info.apply(lambda x:x.sum()))

    for col_name in team_data_columns[6:]:
        handled_team_data[col_name] /= (handled_team_data['上场时间']/60)

    handled_team_data['投篮命中率'] = handled_team_data['投篮命中次数'] / handled_team_data['投篮出手次数']
    handled_team_data['三分命中率'] = handled_team_data['三分命中次数'] / handled_team_data['三分出手次数']
    handled_team_data['罚球命中率'] = handled_team_data['罚球命中次数'] / handled_team_data['罚球出手次数']

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

    # 剔除部分数据
    print(raw_match_data.head(30))
    print(raw_match_data['主场胜负'].value_counts())

    for name in range(208):
        data_temp = raw_match_data[raw_match_data['主场队名'] == name]
        if data_temp.empty:
            rates_of_z.append(0)
        else:
            rates_of_z.append(int(data_temp['主场胜负'].value_counts()[1]) / len(data_temp))
        data_temp = raw_match_data[raw_match_data['客场队名'] == name]
        if data_temp.empty:
            rates_of_k.append(1)
        else:
            rates_of_k.append(int(data_temp['客场胜负'].value_counts()[1]) / len(data_temp))
    
    for z_name in range(208):
        data_temp = raw_match_data[raw_match_data['主场队名']==z_name]
        if data_temp.empty:
            continue
        for k_name in list(data_temp['客场队名']):
            data_temp_t=data_temp[data_temp['客场队名']==k_name]
            #print(data_temp_t['主场胜负'].value_counts()[1])
            rate=0
            try:
                data_temp_t['主场胜负'].value_counts()[1]/len(data_temp_t)
            except:
                pass
            rates_z_to_k.append(rate)


    raw_match_data['主对客历史胜率']=Series(rates_z_to_k)

    print(raw_match_data.head())
    return raw_match_data,rates_of_z,rates_of_k

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

    return raw_team_data

def loadTestData():
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

def write_pred_result(result):
    with open(output_URI, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['主场赢得比赛的置信度'])
        writer.writerows(result)


