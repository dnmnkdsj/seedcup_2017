from data_handle_te import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve,classification_report
from matplotlib import pyplot as plt
from matplotlib import pylab

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier



def plotRUC(y_test,y_train,title=None):
    f_pos,t_pos,thres=roc_curve(y_train,y_test,pos_label=1)
    auc_area=auc(f_pos,t_pos)

    plt.figure(1)
    plt.plot(f_pos,t_pos,'darkorange',lw=2,label='AUC = %.2f'%auc_area)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],color='navy',linestyle='--')
    plt.ylabel('True Pos Rate')
    plt.xlabel('False Pos Rate')
    plt.show()

def plot_pr(auc_score, precision, recall, label=None):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.plot(recall, precision, lw=1)
    pylab.show()

if __name__=='__main__':
    team_features = ['投篮命中率', '投篮命中次数',
                     '投篮出手次数', '三分命中率','三分命中次数','三分出手次数',
                     '罚球命中率','罚球命中次数','罚球出手次数','篮板总数','前场篮板','后场篮板', '助攻',
                     '抢断', '盖帽', '失误', '犯规', '得分','作主场胜率','作客场胜率','等级分']+\
                    ['seed','百回合得分','前胜率','核心球员','DWS','人均进攻篮板比例','人均防守篮板比例','三分比例','罚球比例','百回合失误']
    dataSet,labelSet,testSet = loadDataSet(team_features)
    #print(dataSet.head())
    #dataSet=MinMaxScaler().fit_transform(dataSet)
    #dataSet=VarianceThreshold(threshold=0.01).fit_transform(dataSet)

    dataSet,labelSet=shuffle(dataSet,labelSet)
    print(dataSet.head())

    for i in list(dataSet.columns.values):
        dataSet[i]=dataSet[i].values.reshape(-1,1)



    x_train,x_test,y_train,y_test=train_test_split(dataSet,labelSet,test_size=0.3)

    x_train=dataSet
    y_train=labelSet
    x_test=testSet
    print('load data....')
    print(x_train.head())
    print(x_test.head())

    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

    model = ExtraTreesClassifier()
    model.fit(dataSet, labelSet)
    # 显示每个属性相对重要性
    print(model.feature_importances_)


    #print(np.mean(dataSet))
    #x_train, y_train = shuffle(x_train, y_train)

    #print(x_train[:10])
    print("数据构建完成,开始训练")

    l = lr(penalty='l2',C=1,max_iter=1000,tol=0.00001)
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(l, params, scoring='roc_auc', refit=True)
    clf.fit(x_train, y_train)
    print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
    #l.fit(x_train,y_train)
    result = clf.predict_proba(x_test)

    #print(result[:10])
    #print(y_test[:10])
    single_result = [[x[1]] for x in result]
    re=[x[1] for x in result]
    #print(testSet[44].value_counts())
    #print(roc_auc_score(np.array(y_train),np.array(re)))
    #plotRUC(re,y_train)
    write_pred_result(single_result)

    #precision, recall, thresholds = precision_recall_curve(y_train, re)

    #plot_pr(0.5, precision, recall, "pos")




