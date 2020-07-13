
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt

# 讀資料
df = pd.read_csv('D:/wen/mydata/creditcard.csv')
df = df.sample(frac=1,random_state = 42)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0]
test_df = pd.concat([fraud_df[0:100], non_fraud_df[0:100]]) # 最終測試資料
train_df = pd.concat([fraud_df[100:], non_fraud_df[100:]])
new_df = pd.concat([fraud_df[100:492],non_fraud_df[100:492]])
X = train_df.drop('Class', axis=1)
y = train_df['Class']
nm = NearMiss()
X_res, y_res = nm.fit_resample(X, y)
new2_df = pd.concat([X_res,y_res], axis=1)
new3_df = pd.concat([new_df,new2_df], axis=0)
del [new2_df,new_df,X,y,nm,X_res,y_res]

X = new3_df.drop('Class', axis=1)
y = new3_df['Class']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# 平分訓練、測試資料
from sklearn.model_selection import StratifiedShuffleSplit
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(X, y))
X_train = X.iloc[train_valid_idx]
y_train = y.iloc[train_valid_idx]
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

Amount_scaler_no_mean = StandardScaler(with_mean=False)
Time_scaler_no_mean = StandardScaler(with_mean=False)
Amount_scaler_no_mean.fit(X_train['Amount'].values.reshape(-1,1))
Time_scaler_no_mean.fit(X_train['Time'].values.reshape(-1,1))

scaler = StandardScaler()
scaler.fit(X_train)

Amount_train = Amount_scaler_no_mean.transform(X_train['Amount'].values.reshape(-1,1))
Time_train = Time_scaler_no_mean.transform(X_train['Time'].values.reshape(-1,1))
X_train = scaler.transform(X_train)
X_train[:,0] = Time_train.reshape(-1,)
X_train[:,-1] = Amount_train.reshape(-1,)

Amount_test = Amount_scaler_no_mean.transform(X_test['Amount'].values.reshape(-1,1))
Time_test = Time_scaler_no_mean.transform(X_test['Time'].values.reshape(-1,1))
X_test = scaler.transform(X_test)
X_test[:,0] = Time_test.reshape(-1,)
X_test[:,-1] = Amount_test.reshape(-1,)


# SVC
svc_params = {'C': [0.1, 1, 5, 10, 50, 100], 
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
              'class_weight':['balanced',None]}
grid_svc = GridSearchCV(SVC(probability=True), svc_params, n_jobs=-1)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_

print('The best estimator SVC')
confusion_matrix(y_train, svc.predict(X_train))
confusion_matrix(y_test, svc.predict(X_test))
f1_score(y_test, svc.predict(X_test), average='weighted')


# Bagging
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(svc, n_estimators = 20)
bag_clf.fit(X_train, y_train)

confusion_matrix(y_train, bag_clf.predict(X_train))
confusion_matrix(y_test, bag_clf.predict(X_test))
f1_score(y_test,  bag_clf.predict(X_test), average='weighted')


# Ada boost (DecisionTree)
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
                             n_estimators = 100,
                             learning_rate = 0.5)

ada_clf.fit(X_train, y_train)
confusion_matrix(y_train, ada_clf.predict(X_train))
confusion_matrix(y_test, ada_clf.predict(X_test))
f1_score(y_test,  ada_clf.predict(X_test), average='weighted')



# GradientBoost
from sklearn.ensemble import GradientBoostingClassifier
gradientboost_clf = GradientBoostingClassifier(max_depth = 4,
                                               subsample = 1)
gradientboost_clf.fit(X_train, y_train)

confusion_matrix(y_train, gradientboost_clf.predict(X_train))
confusion_matrix(y_test, gradientboost_clf.predict(X_test))
f1_score(y_test,  gradientboost_clf.predict(X_test), average='weighted')



# Xgboost
from xgboost import XGBClassifier

gbm_param_grid = {
    #'booster' : [ 'dart','gbtree'],
    'n_estimators': [70,100], # 樹有幾棵
    'max_depth': [4,6,8,10], # 樹的深度
    'learning_rate': [0.2, 0.4, 0.5, 0.6],
    'colsample_bytree': [0.5, 0.75, 1]
    }

xgb = GridSearchCV(XGBClassifier(),
                   gbm_param_grid,
                   verbose=1, 
                   n_jobs=-1,
                   scoring ='f1_weighted')
xgb.fit(X_train, y_train)
xgb_best = xgb.best_estimator_

print('The best estimator xgb')
confusion_matrix(y_train, xgb_best.predict(X_train))
confusion_matrix(y_test, xgb_best.predict(X_test))
f1_score(y_test, xgb_best.predict(X_test), average='weighted')

a = xgb.cv_results_


from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator,estimator_name):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    f, ax = plt.subplots(1,1, figsize=(6,6))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Mean Test score")
    ax.set_title(estimator_name, fontsize=14)
    ax.set_xlabel('Training size (m)')
    ax.set_ylabel('f1 Score')
    ax.grid(True)
    ax.legend(loc="best")
    return plt

plot_learning_curve(svc,'svc')
plot_learning_curve(bag_clf,'Bagging')
plot_learning_curve(ada_clf,'AdaBoost')
plot_learning_curve(gradientboost_clf,'GradientBoost')
plot_learning_curve(xgb_best,'Xgboost')



