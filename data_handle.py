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

base_dir=os.path.abspath(os.path.dirname(__file__))
match_data_URI=os.path.join(base_dir,'data/matchDataTrain.csv')
team_data_URI=os.path.join(base_dir,'data/teamData.csv')





def loadDataSet():
    raw_team_data=loadTeamData()
    raw_match_data=loadMatchData()

    team_data_columns=list(raw_team_data.columns.values)
    #print(team_data_columns)
    for col_name in team_data_columns[4:]:
        raw_team_data[col_name]*=raw_team_data['出场次数']
    


    print(raw_team_data.head())



def loadMatchData():
    '''
    :return:
    raw_match_data为pandas内置的DataFrame类型
    列标签为 （import 然后运行可知）
    '''
    raw_match_data=pd.read_csv(match_data_URI)

    cols_to_change=["客场本场前战绩","主场本场前战绩","比分（客场:主场）"]

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
        raw_match_data.pop(col_name)

    for frame in [dataframe_temp1,dataframe_temp2,dataframe_temp3]:
        for colname in list(frame.columns.values):
            raw_match_data[colname]=frame[colname]

    return raw_match_data

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


def loadMatchData1():
    '''
    抛弃

    :return:
    [
        [180,138,0,1,0,0,128,132],
        [145,138,0,1,1,0,91,109],
        ...
    ]
    '''

    re_match_data=[]

    with open(match_data_URI,'r') as f:
        f_csv=csv.reader(f)
        headers=next(f_csv)
        print(headers)
        for row in f_csv:
            re_match_data.append([])
            #添加客场队名 int
            re_match_data[-1].append(int(row[0]))
            #添加主场队名 int
            re_match_data[-1].append(int(row[1]))
            #添加客场前战绩  胜，负 int
            row[2].replace("胜"," ")
            row[2].replace("负", " ")
            re_match_data[-1].extend([int(s) for s in row[2].split()])
            #添加主场前战绩  胜，负 int
            row[3].replace("胜", " ")
            row[3].replace("负", " ")
            re_match_data[-1].extend([int(s) for s in row[3].split()])
            #添加比分  客，主 int
            row[4].replace(":"," ")
            re_match_data[-1].extend([int(s) for s in row[4].split()])
        f.close()

    return re_match_data

def loadTeamData1():
    '''
    抛弃

    :return:
    [
        [
            [],[],...
        ],
        ...
    ]
    exp: re_team_data[0][1][2] 代表队名为0的队伍中1号队员的第二项数据
    PS:所有数据均读取，且读取为int类型
    '''

    re_team_data=[]

    with open(team_data_URI,'r') as f:
        f_csv=csv.reader(f)
        headers=next(f_csv)

        for row in f_csv:
            for i in range(len(row)):
                row[i]=float(row[i])

            if row[0]>=len(re_team_data):
                re_team_data.append([])

            re_team_data[row[0]].append(row)

        return re_team_data








