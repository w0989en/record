#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 資料來源、參考網站：https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets?select=creditcard.csv#notebook-container
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# 載入資料
import pandas as pd
import numpy as np
df = pd.read_csv('D:/wen/mydata/creditcard.csv')
df.head()


# In[3]:


# 敘述統計
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot('Class', data=df)
plt.title('Class\n(0: No Fraud || 1: Fraud)', fontsize=14)


# In[4]:


# [Amount]的機率密度圖
ax = sns.distplot(df['Amount'].values,  color='r')
ax.set_title('Amount density plot', fontsize=14)
ax.set_xlim([min(df['Amount'].values), max(df['Amount'].values)])
plt.show()


# In[5]:


#[Time]的機率密度圖
ax = sns.distplot(df['Time'].values, color='b')
ax.set_title('Time density plot', fontsize=14)
ax.set_xlim([min(df['Time'].values), max(df['Time'].values)])
plt.show()


# In[6]:


# rescale [Amount],[Time]
from sklearn.preprocessing import StandardScaler, RobustScaler
std_scaler = StandardScaler(with_mean=False)
rob_scaler = RobustScaler()
df['Amount'] = std_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = std_scaler.fit_transform(df['Time'].values.reshape(-1,1))


# In[7]:


# 打亂順序
df = df.sample(frac=1,random_state = 42)

# undersampling (平衡類別)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0]
test_df = pd.concat([fraud_df[0:100], non_fraud_df[0:100]]) # 最終測試資料
train_df = pd.concat([fraud_df[100:], non_fraud_df[100:]])

new_df = pd.concat([fraud_df[100:492],non_fraud_df[100:492]])


# In[8]:


# 平衡後的資料 
sns.countplot('Class', data=new_df)
plt.title('Class\n(0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()


# In[9]:


# 檢視其餘參數對上類別 (平衡後資料)
for i in range(7):
    f, axes = plt.subplots(2, 2, figsize=(12,10))
    sns.boxplot(x="Class", y=f"V{i*4+1}", data=new_df, ax=axes[0][0])
    axes[0][0].set_title(f'V{i*4+1} vs Class Negative Correlation')
    sns.boxplot(x="Class", y=f"V{i*4+2}", data=new_df, ax=axes[0][1])
    axes[0][1].set_title(f'V{i*4+2} vs Class Negative Correlation')
    sns.boxplot(x="Class", y=f"V{i*4+3}", data=new_df, ax=axes[1][0])
    axes[1][0].set_title(f'V{i*4+3} vs Class Negative Correlation')
    sns.boxplot(x="Class", y=f"V{i*4+4}", data=new_df, ax=axes[1][1])
    axes[1][1].set_title(f'V{i*4+4} vs Class Negative Correlation')
    plt.show()


# In[10]:


# correlation matrix(全資料)
corr = df.corr()
f, ax = plt.subplots(1,1, figsize=(14,10))
ax = sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20})
ax.set_title("correlation matrix", fontsize=14)


# In[11]:


# 檢視Class分佈圖，t-SNE, PCA, SVD (平衡後資料)
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches

X = new_df.drop('Class', axis=1)
y = new_df['Class']

# T-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized').fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))


# In[12]:


# t-SNE
s_1 = [0]*len(y) # y=1:Fraud
for i in range(len(y)):
    if y.iloc[i] == 1:
        s_1[i] = 30
s_0 = s_0 = [30 - x for x in s_1] # Not Fraud
blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c='#8a0000', s=s_1, cmap='coolwarm', label='Fraud', linewidths=2, alpha=0.5)
plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c='#164ec7', s=s_0, cmap='coolwarm', label='Not Fraud', linewidths=2, alpha=0.5)
plt.grid(True)
plt.title("t-SNE")
plt.legend(handles=[blue_patch, red_patch])


# In[13]:


# PCA, SVD
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))

ax1.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c='#8a0000', s=s_1, cmap='coolwarm', label='Fraud', linewidths=2, alpha=0.5)
ax1.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c='#164ec7', s=s_0, cmap='coolwarm', label='Not Fraud', linewidths=2, alpha=0.5)
ax1.grid(True)
ax1.set_title("PCA")
ax1.legend(handles=[blue_patch, red_patch])

ax2.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c='#8a0000', s=s_1, cmap='coolwarm', label='Fraud', linewidths=2, alpha=0.5)
ax2.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c='#164ec7', s=s_0, cmap='coolwarm', label='Not Fraud', linewidths=2, alpha=0.5)
ax2.grid(True)
ax2.set_title("Truncated SVD")
ax2.legend(handles=[blue_patch, red_patch])


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


# 訓練模型，根據自訂函數選擇最佳模型，自動選擇最佳參數 (平衡後資料)(train data)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer
import numpy as np


# 自訂選模函數
def my_scoring_fun(y_true, y_predict):
    sc = np.sum(np.maximum(y_true - y_predict, 0))
    return sc
my_scorer = make_scorer(my_scoring_fun, greater_is_better=False)

# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, scoring=my_scorer, n_jobs=-1)
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_

# KNears
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params, scoring=my_scorer, n_jobs=-1)
grid_knears.fit(X_train, y_train)
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.1, 0.5, 1, 5, 10, 50, 90], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
              'class_weight':['balanced',None]}
grid_svc = GridSearchCV(SVC(), svc_params, scoring=my_scorer, n_jobs=-1)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_

# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(4,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params, scoring=my_scorer, n_jobs=-1)
grid_tree.fit(X_train, y_train)
tree_clf = grid_tree.best_estimator_

# RandomForestClassifier Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(4,7,1))}
RandomForest = GridSearchCV(RandomForestClassifier(), tree_params, scoring=my_scorer, n_jobs=-1)
RandomForest.fit(X_train, y_train)
RandomForest_clf = RandomForest.best_estimator_


# In[16]:


# F1分數，交叉驗證 (平衡後資料)(train data)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=cv, scoring='f1')
print('Logistic Regression F1 Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors,X_train, y_train, cv=cv, scoring='f1')
print('Knears Neighbors F1 Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=cv, scoring='f1')
print('Support Vector Classifier F1 Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=cv, scoring='f1')
print('DecisionTree Classifier F1 Score', round(tree_score.mean() * 100, 2).astype(str) + '%')

RandomForest_score = cross_val_score(RandomForest_clf, X_train, y_train, cv=cv, scoring='f1')
print('RandomForest Classifier F1 Score', round(RandomForest_score.mean() * 100, 2).astype(str) + '%')


# In[17]:


# learning rate (平衡後資料)(train data)
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
    ax.set_ylabel('Score')
    ax.grid(True)
    ax.legend(loc="best")
    return plt

plot_learning_curve(log_reg,'Logistic Regression')
plot_learning_curve(knears_neighbors,'knears_neighbors')
plot_learning_curve(svc,'svc')
plot_learning_curve(tree_clf,'tree')
plot_learning_curve(RandomForest_clf,'RandomForest')


# In[18]:


# 預測 cross_val_predict(平衡後資料)(train data)
from sklearn.model_selection import cross_val_predict
log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")
knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)
svc_pred = cross_val_predict(svc, X_train, y_train, cv=5, method="decision_function")
tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)
RandomForest_pred = cross_val_predict(RandomForest_clf, X_train, y_train, cv=5)


# In[19]:


# roc curve (平衡後資料)(train data)
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

log_fpr, log_tpr, log_threshold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)
RandomForest_fpr, RandomForest_tpr, RandomForest_threshold = roc_curve(y_train, RandomForest_pred)

plt.figure(figsize=(10,6))
plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
plt.plot(RandomForest_fpr, RandomForest_tpr, label='RandomForest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, RandomForest_pred)))
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.01, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend()

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))
print('RandomForest Classifier: ', roc_auc_score(y_train, RandomForest_pred))


# In[20]:


# 預測 (平衡後資料)(test data)
y_pred_log_reg = log_reg.predict(X_test)
y_pred_knear = knears_neighbors.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)
y_pred_RandomForest = RandomForest_clf.predict(X_test)


# In[21]:


# 混淆矩陣 Confusion matrix [真實分類(y軸)\預測分類(x軸)]
from sklearn.metrics import confusion_matrix 
log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
svc_cf = confusion_matrix(y_test, y_pred_svc)
tree_cf = confusion_matrix(y_test, y_pred_tree)
RandomForest_cf = confusion_matrix(y_test, y_pred_RandomForest)


# In[22]:


def plot_confusion_matrix(estimator, title):
    f, ax = plt.subplots(1,1, figsize=(8,6))
    ax = sns.heatmap(estimator, annot=True, cmap=plt.cm.copper)
    ax.set_title(f"{title} \n Confusion Matrix", fontsize=14)
    ax.set_xticklabels(['Not Fraud', 'Fraud'], fontsize=12, rotation=0)
    ax.set_yticklabels(['Not Fraud', 'Fraud'], fontsize=12, rotation=360)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    return plt
    
plot_confusion_matrix(log_reg_cf, "Logistic Regression")
plot_confusion_matrix(kneighbors_cf, "KNearsNeighbors")
plot_confusion_matrix(svc_cf, "SVC")
plot_confusion_matrix(tree_cf, "DecisionTree Classifier")
plot_confusion_matrix(RandomForest_cf, "RandomForest Classifier")


# In[23]:


# 顯示預測結果 (平衡後資料)(test data)
from sklearn.metrics import classification_report
print('Logistic Regression:')
print(classification_report(y_test, y_pred_log_reg))
print('KNears Neighbors:')
print(classification_report(y_test, y_pred_knear))
print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_svc))
print('DecisionTree Classifier:')
print(classification_report(y_test, y_pred_tree))
print('RandomForest Classifier:')
print(classification_report(y_test, y_pred_RandomForest))


# In[24]:


# 神經網路預測 (平衡後資料)(train data)
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]
nn_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')
])
#nn_model.summary()
nn_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=0)


# In[25]:


nn_predictions = nn_model.predict_classes(X_test, batch_size=200, verbose=0)
nn_cm = confusion_matrix(y_test, nn_predictions)
plot_confusion_matrix(nn_cm, "nn Classifier")
print('nn Classifier:')
print(classification_report(y_test, nn_predictions))


# 綜合上面圖表，SVC的表現最好，而且SVC還有其他可調參數，當訓練資料上升時，F1分數可能更好。

# In[28]:


# 檢視SVC的預測情況 (平衡後資料)(train data)
p_d_f = svc.decision_function(X_train)

zippedList =  list(zip(y_train, y_pred_svc, p_d_f))
df_svc = pd.DataFrame(zippedList, columns = ['true_y','predict_y','decision']) 
df_svc['index'] = df_svc.index.values
df_svc['false'] = list(map(int, df_svc['true_y'] == df_svc['predict_y']))

f, ax = plt.subplots(1,1, figsize=(10,6))
ax.scatter(df_svc['index'], df_svc['decision'], c='#164ec7', s=df_svc['false']*20, cmap='coolwarm', label='predict correct', alpha=1)
ax.scatter(df_svc['index'], df_svc['decision'], c='#8a0000', s=(1-df_svc['false'])*20, cmap='coolwarm', label='predict False', alpha=1)
ax.set_ylim(-7,30)
ax.axhspan(0, 30, facecolor='green', alpha=0.15, label='predict to Fraud')
ax.axhspan(-7, 0, facecolor='yellow', alpha=0.15, label='predict to No Fraud')
plt.legend(fancybox=True, framealpha=1, borderpad=1, loc = 'best')
plt.ylabel('decision', fontsize=12)


# 多數的錯誤(紅點)是把Fraut分成No Fraut，可能是與訓練資料有關

# In[53]:


# under-sampling : NearMiss3 
from imblearn.under_sampling import NearMiss
X = train_df.drop('Class', axis=1)
y = train_df['Class']
nm = NearMiss()
X_res, y_res = nm.fit_resample(X, y)


# In[54]:


# under-sampling : 建模(train data)、預測(test data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
svc_params = {'C': [0.1, 0.5, 1, 5, 10, 50, 90], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
              'class_weight':['balanced',None]}
grid_svc = GridSearchCV(SVC(), svc_params, n_jobs=-1)
grid_svc.fit(X_train, y_train)
svc2 = grid_svc.best_estimator_

y_pred_svc = svc2.predict(X_test)
svc_cf2 = confusion_matrix(y_test, y_pred_svc)
print(svc_cf2)
plot_confusion_matrix(svc_cf2, "under-sampling SVC Classifier")


# In[56]:


# under-sampling SVC 最終測試資料 (test_df) 
y_pred_svc_ALL = svc2.predict(test_df.iloc[:,0:-1])
svc_cf_ALL2 = confusion_matrix(test_df['Class'], y_pred_svc_ALL)
print(svc_cf_ALL2)
plot_confusion_matrix(svc_cf_ALL2, "under-sampling SVC Classifier")
print('Support Vector Classifier:')
print(classification_report(test_df['Class'], y_pred_svc_ALL))


# In[47]:


# 重新安排data，混合上面2組data，使得訓練要本上升
new2_df = pd.concat([X_res,y_res], axis=1)
new3_df = pd.concat([new_df,new2_df], axis=0)
X_train, X_test, y_train, y_test = train_test_split(new3_df.iloc[:,0:30], new3_df.iloc[:,30], test_size=0.2, random_state=42)


# In[59]:


# 仔細調整參數，且在預測(test data)，有較好的結果
svc_params = {'C': [1, 10, 50, 90, 120, 180], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
              'class_weight':['balanced',None], 'gamma':['scale', 'auto', 0.1, 1, 5, 10, 20] }
grid_svc = GridSearchCV(SVC(), svc_params, n_jobs=-1)
grid_svc.fit(X_train, y_train)
svc3 = grid_svc.best_estimator_
print(svc3)
y_pred_svc = svc3.predict(X_test)
svc_cf3 = confusion_matrix(y_test, y_pred_svc)
print(svc_cf3)
plot_confusion_matrix(svc_cf3, "combined data SVC Classifier")


# In[60]:


plot_learning_curve(svc3,'combined data SVC')


# In[61]:


# 最終測試資料 (test_df)
y_pred_svc_ALL = svc3.predict(test_df.iloc[:,0:-1])
svc_cf_ALL3 = confusion_matrix(test_df['Class'], y_pred_svc_ALL)
print(svc_cf_ALL3)
plot_confusion_matrix(svc_cf_ALL3, "combined data SVC Classifier")
print('Support Vector Classifier:')
print(classification_report(test_df['Class'], y_pred_svc_ALL))




