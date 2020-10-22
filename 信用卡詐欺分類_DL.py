
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
df = pd.read_csv('D:/wen/mydata/creditcard.csv')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
df = df.sample(frac=1,random_state = 42)

from sklearn.preprocessing import StandardScaler, RobustScaler
std_scaler = StandardScaler(with_mean=False)
#std_scaler = StandardScaler()
df['Amount'] = std_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = std_scaler.fit_transform(df['Time'].values.reshape(-1,1))


# undersampling (平衡類別)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0]
test_df = pd.concat([fraud_df[0:100], non_fraud_df[0:100]]) # 最終測試資料
train_df = pd.concat([fraud_df[100:], non_fraud_df[100:]])
new_df = pd.concat([fraud_df[100:492],non_fraud_df[100:492]])


X = new_df.drop('Class', axis=1)
y = new_df['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############################################
## keras方法
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
history = nn_model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=100, shuffle=True)

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

nn_predictions = nn_model.predict_classes(X_test, batch_size=200, verbose=0)
confusion_matrix(y_test, nn_predictions)

############################################################
#  TF.keras方法
# CH 10
import tensorflow as tf
tf.__version__
from tensorflow import keras


def build_model(n_hidden=1, n_neurons=30, learning_rate=2e-3, 
                input_shape=X_train.shape[1]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(2, activation='softmax'))
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer=optimizer, metrics=['accuracy'])
    return model

keras_reg = keras.wrappers.scikit_learn.KerasClassifier(build_model)

keras_reg.fit(X_train, y_train, validation_split=0.2, batch_size=128, 
              epochs=100, shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])


from sklearn.metrics import confusion_matrix 
nn_predictions = keras_reg.predict(X_test)
confusion_matrix(y_test, nn_predictions)


##################################################################
#  TF.keras方法 + GridSearchCV 

from sklearn.model_selection import GridSearchCV 

param_distribs = {
    "n_hidden": [ 1, 2, 3],
    "n_neurons": [ 10, 30, 50],
    "learning_rate": [ 0.001, 0.002, 0.003]
}

keras_reg = keras.wrappers.scikit_learn.KerasClassifier(build_model)

rnd_search_cv = GridSearchCV(keras_reg, param_distribs, cv=3, 
                             verbose=2, n_jobs=-1)

rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_split=0.2,
                  batch_size=128,
                  shuffle=True,
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])


print(rnd_search_cv.best_params_)
model = rnd_search_cv.best_estimator_
# model = rnd_search_cv.best_estimator_.model #連續資料用

history = model.fit(X_train, y_train, validation_split=0.2, batch_size=128, 
              epochs=100, shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])

# plot loss and accurace
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# confusion matrix
nn_predictions = model.predict(X_test)
confusion_matrix(y_test, nn_predictions)

########################################
########################################
# CH 11

n_inputs = X_train.shape[1]
nn_model = keras.models.Sequential([
    keras.layers.Dense(n_inputs, input_shape=(n_inputs, ),
                       kernel_initializer="he_normal",
                       kernel_constraint=keras.constraints.max_norm(1.)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.3),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(16, kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(16, kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(8, kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Activation('softmax')
])


nn_model.compile(keras.optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train, y_train, validation_split=0.2, batch_size=64, 
                       epochs=200, shuffle=True,
                       callbacks=[keras.callbacks.EarlyStopping(patience=5)])

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


nn_predictions = nn_model.predict_classes(X_test, batch_size=200, verbose=0)
confusion_matrix(y_test, nn_predictions)











