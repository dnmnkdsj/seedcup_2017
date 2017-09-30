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
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
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

if __name__=='__main__':
    team_features = ['投篮命中率', '投篮命中次数',
                     '投篮出手次数', '三分命中率','三分命中次数','三分出手次数',
                     '罚球命中率','罚球命中次数','罚球出手次数','篮板总数','前场篮板','后场篮板', '助攻',
                     '抢断', '盖帽', '失误', '犯规', '得分','作主场胜率','作客场胜率','等级分']+\
                    ['上场时间','攻防比','失误比']
    dataSet,labelSet,testSet = loadDataSet(['作主场胜率','作客场胜率','投篮命中率','投篮命中次数','投篮出手次数'])
    #print(dataSet.head())
    #dataSet=MinMaxScaler().fit_transform(dataSet)
    #dataSet=VarianceThreshold(threshold=0.01).fit_transform(dataSet)

    dataSet,labelSet=shuffle(dataSet,labelSet)
    print(dataSet.head())

    #for i in list(dataSet.columns.values):
    #    dataSet[i]=dataSet[i].values.reshape(-1,1)



    x_train,x_test,y_train,y_test=train_test_split(dataSet,labelSet,test_size=0.2)
    '''
    x_train=dataSet
    y_train=labelSet
    x_test=testSet
    print('load data....')
    print(x_train.head())
    print(x_test.head())
    '''
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

    l = lr(penalty='l1',C=0.01,max_iter=1000,tol=0.000001)
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(l, params, scoring='roc_auc', refit=True)
    clf.fit(x_train, y_train)
    print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
    #l.fit(x_train,y_train)
    result = clf.predict_proba(x_train)

    #print(result[:10])
    #print(y_test[:10])
    single_result = [[x[1]] for x in result]
    re=[x[1] for x in result]
    #print(testSet[44].value_counts())
    print(re[:30])
    print(roc_auc_score(np.array(y_train),np.array(re)))
    #plotRUC(re,y_test)
    #write_pred_result(single_result)
    '''
    print('---------test----------')

    gbm0 = GradientBoostingClassifier(random_state=10)
    gbm0.fit(x_train, y_train)
    y_pred = gbm0.predict(x_test)
    y_predprob = gbm0.predict_proba(x_test)[:, 1]
    print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))
    
    param_test1 = {'n_estimators': range(2, 20, 2)}
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                 min_samples_leaf=20, max_depth=8, max_features='sqrt',
                                                                 subsample=0.8, random_state=10),
                            param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    gsearch1.fit(x_test, y_test)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    param_test2 = {'max_depth': range(3, 14, 1), 'min_samples_split': range(100, 801, 100)}
    gsearch2 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, min_samples_leaf=20,
                                             max_features='sqrt', subsample=0.8, random_state=10),
        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    gsearch2.fit(x_test, y_test)
    print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
    
    param_test3 = {'min_samples_split': range(20, 200, 40), 'min_samples_leaf': range(80, 141, 10)}
    gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, max_depth=5,
                                                                 max_features='sqrt', subsample=0.8, random_state=10),
                            param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    gsearch3.fit(x_test, y_test)
    print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
    
    '''
    gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=5, min_samples_leaf=60,
                                      min_samples_split=100, max_features='sqrt', subsample=0.8, random_state=10)
    gbm1.fit(x_train, y_train)
    y_pred = gbm1.predict(x_test)
    y_predprob = gbm1.predict_proba(x_test)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))
    '''
    param_test4 = {'max_features': range(7, 44, 2)}
    gsearch4 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=5, min_samples_leaf=60,
                                             min_samples_split=100, subsample=0.8, random_state=10),
        param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
    gsearch4.fit(x_train, y_train)
    print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

    param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    gsearch5 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=5, min_samples_leaf=60,
                                             min_samples_split=100, max_features=27, random_state=10),
        param_grid=param_test5, scoring='roc_auc', iid=False, cv=5)
    gsearch5.fit(x_train, y_train)
    print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)
    '''
    gbm1 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=200, max_depth=5, min_samples_leaf=60,
                                      min_samples_split=100, max_features='sqrt', subsample=0.75, random_state=10)
    gbm1.fit(x_train, y_train)
    y_pred = gbm1.predict(x_test)
    y_predprob = gbm1.predict_proba(x_test)[:, 1]
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))

    gbm1_enc=OneHotEncoder()
    gbm1_lm=lr()

    x_train,x_train_lr,y_train,y_train_lr=train_test_split(x_train,y_train,test_size=0.5)

    gbm1.fit(x_train,y_train)
    gbm1_enc.fit(gbm1.apply(x_train)[:,:,0])

    gbm1_lm.fit(gbm1_enc.transform(gbm1.apply(x_train_lr)[:,:,0]),y_train_lr)

    y_pred_gbm1_lm=gbm1_lm.predict_proba(gbm1_enc.transform(gbm1.apply(x_test)[:,:,0]))[:,1]
    print(y_pred_gbm1_lm[:3])
    print(roc_auc_score(y_test,y_pred_gbm1_lm))

