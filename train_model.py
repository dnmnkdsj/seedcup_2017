from data_handle_te import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV as lr
from sklearn.metrics import roc_auc_score,roc_curve,auc
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


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

if __name__=='__main__':
    team_features = ['投篮命中率', '投篮命中次数',
                     '投篮出手次数', '三分命中率','三分命中次数','三分出手次数',
                     '罚球命中率','罚球命中次数','罚球出手次数','篮板总数','前场篮板','后场篮板', '助攻',
                     '抢断', '盖帽', '失误', '犯规', '得分']
    dataSet,labelSet = loadDataSet(team_features)

    #dataSet=MinMaxScaler().fit_transform(dataSet)
    #dataSet=VarianceThreshold(threshold=0.01).fit_transform(dataSet)
    print(np.mean(dataSet))

    x_train,x_test,y_train,y_test=train_test_split(dataSet,labelSet,test_size=0.1)
    """
    sc = StandardScaler()
    sc.fit(x_train)
    print(sc.mean_)
    print(sc.scale_)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    """


    print(x_train[:10])
    print("数据构建完成,开始训练")
    l = lr(penalty='l2',random_state=0,class_weight='balanced',max_iter=200,tol=0.0000001)
    #params = {'C': np.logspace(start=-5, stop=3, num=9)}
    #clf = GridSearchCV(l, params, scoring='neg_log_loss', refit=True)
    #clf.fit(x_train, y_train)
    #print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
    l.fit(x_train,y_train)
    result = l.predict_proba(x_test)

    print(result[:30])
    print(y_test[:30])
    single_result = [[x[1]] for x in result]
    re=[x[1] for x in result]

    print(roc_auc_score(np.array(y_test),np.array(re)))
    #plotRUC(re,y_test)
    #write_pred_result(single_result)

