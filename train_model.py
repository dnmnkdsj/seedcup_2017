from data_handle_te import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import roc_auc_score,roc_curve,auc
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import cross_validation, metrics
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
                     '抢断', '盖帽', '失误', '犯规', '得分','作主场胜率','作客场胜率']
    dataSet,labelSet,testSet = loadDataSet(team_features)
    #dataSet=MinMaxScaler().fit_transform(dataSet)
    #dataSet=VarianceThreshold(threshold=0.01).fit_transform(dataSet)

    for i in list(dataSet.columns.values):
        dataSet[i]=dataSet[i].values.reshape(-1,1)



    x_train,x_test,y_train,y_test=train_test_split(dataSet,labelSet,test_size=0.2)

    #x_train=dataSet
    #y_train=labelSet
    #x_test=testSet

    #print(x_train)
    #print(x_test)

    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)


    print(np.mean(dataSet))
    x_train, y_train = shuffle(x_train, y_train)

    print(x_train[:10])
    print("数据构建完成,开始训练")

    l = lr()
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(l, params, scoring='roc_auc', refit=True)
    clf.fit(x_train, y_train)
    print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
    #l.fit(x_train,y_train)
    result = clf.predict_proba(x_test)

    print(result[:10])
    print(y_test[:10])
    single_result = [[x[1]] for x in result]
    re=[x[1] for x in result]
    #print(testSet[44].value_counts())

    print(roc_auc_score(np.array(y_test),np.array(re)))
    #plotRUC(re,y_test)
    #write_pred_result(single_result)
    '''
    print('--------------------')

    gbm0 = GradientBoostingClassifier(random_state=10)
    gbm0.fit(x_train, y_train)
    y_pred = gbm0.predict(x_train)
    y_predprob = gbm0.predict_proba(x_train)[:, 1]
    print ("Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))

    param_test1 = {'n_estimators': range(20, 81, 10)}
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                 min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                                 subsample=0.8, random_state=10),
                            param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    gsearch1.fit(x_test, y_test)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
    gsearch2 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=40, min_samples_leaf=20,
                                             max_features='sqrt', subsample=0.8, random_state=10),
        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    gsearch2.fit(x_test, y_test)
    gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

    '''
