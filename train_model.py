from data_handle import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import roc_auc_score,roc_curve,auc
from matplotlib import pyplot as plt


def plotRUC(y_train,y_test,title=None):
    f_pos,t_pos,thres=roc_curve(y_train,y_test)
    print(f_pos)
    auc_area=auc(f_pos,t_pos)

    plt.figure(1)
    plt.plot(f_pos,t_pos,'darkorange',lw=2,label='AUC = %.2f'%auc_area)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],color='navy',linestyle='--')
    plt.ylabel('True Pos Rate')
    plt.xlabel('False Pos Rate')
    plt.show()

if __name__=='__main__':
    team_data, feature_data, label = loadDataSet()
    #print(len(feature_data))
    #print(len(label))
    #feature_data=np.mat(feature_data)
    #label=np.mat(label)[0]
    #print(feature_data)
    #print(label)
    x_train,x_test,y_train,y_test=train_test_split(feature_data,label)
    print(x_train[:3])
    print("数据构建完成,开始训练")
    l = lr()
    l.fit(x_train, y_train)
    result = l.predict_proba(x_test)
    print(y_test)

    single_result = [[x[0]] for x in result]
    re=[x[0] for x in result]
    print(roc_auc_score(np.array(y_test),np.array(re)))
    plotRUC(re,y_test)
    write_pred_result(single_result)

