#!/usr/bin/env python
# coding: utf-8

# # Detecting Purchasing Probability in Real-Time using Clickstream Data 

# ## Load Libraries

# In[ ]:


get_ipython().system('pip install --upgrade -q scikeras')
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor,KerasClassifier
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense,LSTM, SimpleRNN, GRU, Dropout
from keras.layers import *
from keras import callbacks
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import joblib
import dill
import time


# ## Load and PreProcess Data

# In[ ]:


upload_data = pd.read_csv('thesis_data.csv')


# In[ ]:


def cleaning_data (csv):
    df = csv.drop(columns=['sessionid'])
    df = df.sample(frac=1).reset_index(drop=True)
    df['converted'] = df['converted'].replace(np.nan, 0)
    df['converted'] = df['converted'].replace("yes", 1) #we need to define it as 0 or 1 to calculate probability
    df['eventlist'] = df['eventstring'].str.strip('[]').str.replace('\'','').str.split(', ')
    df.eventlist = df.eventlist.apply(lambda x: [elem for elem in x if 'order' not in elem])
    df = df[df['date'].notna()]
    df = df.dropna()
    return df

data = cleaning_data(upload_data)
print(len(data))
print("Number of products bought, yes/no:")
data['converted'].value_counts()


# In[ ]:


data.isnull().sum(axis = 0)


# In[ ]:


data.head()


# In[ ]:


data['device'].value_counts()


# In[ ]:


data.device.value_counts().sort_values().plot(kind = 'barh')


# In[ ]:


data['custype'].value_counts()


# In[ ]:


data.custype.value_counts().sort_values().plot(kind = 'barh')


# In[ ]:


## We need to clean the eventlist items so the program can count distinct
def to_1D(series):
    return pd.Series([x for _list in series for x in _list])


# In[ ]:


fig, ax = plt.subplots(figsize = (14,4))
ax.bar(to_1D(data["eventlist"]).value_counts().index,
        to_1D(data["eventlist"]).value_counts().values)
ax.set_ylabel("Frequency", size = 12)
ax.set_title("Event", size = 14)
plt.xticks(rotation = 90) 
plt.show()


# In[ ]:


## Convert Event String into List and count the different types of events, and the length of the event list.
#We will ignore order since this is the same as converted
data['eventlist_len'] = data['eventlist'].str.len()
data['eventlist_add_to_cart'] = data['eventstring'].str.count('add_to_cart')
data['eventlist_homepage_view'] = data['eventstring'].str.count('homepage_view')
data['eventlist_product_view'] = data['eventstring'].str.count('product_view')
data['eventlist_list_impression'] = data['eventstring'].str.count('list_impression')
data['eventlist_remove_from_cart'] = data['eventstring'].str.count('remove_from_cart')
data['eventlist_basket_open'] = data['eventstring'].str.count('basket_open')
data['eventlist_product_click'] = data['eventstring'].str.count('product_click')
data['eventlist_clear_basket'] = data['eventstring'].str.count('clear_basket')
data['eventlist_add_all_to_cart'] = data['eventstring'].str.count('add_all_to_cart')


PadLength = data['eventlist_len'].max()


event_enum = ['add_to_cart', 'homepage_view', 'product_view', 'list_impression', 'remove_from_cart', 'basket_open', 'product_click', 'clear_basket', 'add_all_to_cart']
def mapper(x):
    for ix, i in enumerate(event_enum):
        if i in x:
            return ix+1
    return 0

## Encode the eventList into a vector of Numbers
data['eventlist_coded'] = data['eventlist'].apply(lambda x: [mapper(i) for i in x])


# In[ ]:


print(PadLength)


# In[ ]:


plt.boxplot((data["eventlist_len"]), notch=None, vert=None, patch_artist=None, widths=None)


# In[ ]:


counting = data[data['eventlist_len'] < 7500]
print(len(counting))


# In[ ]:


plt.boxplot((counting["eventlist_len"]), notch=None, vert=None, patch_artist=None, widths=None)


# In[ ]:


data = data[data['eventlist_len'] <= 1500]
print(len(data))


# In[ ]:


## Convert Event String into List and count the different types of events, and the length of the event list.
#We will ignore order since this is the same as converted
data['eventlist_len'] = data['eventlist'].str.len()
data['eventlist_add_to_cart'] = data['eventstring'].str.count('add_to_cart')
data['eventlist_homepage_view'] = data['eventstring'].str.count('homepage_view')
data['eventlist_product_view'] = data['eventstring'].str.count('product_view')
data['eventlist_list_impression'] = data['eventstring'].str.count('list_impression')
data['eventlist_remove_from_cart'] = data['eventstring'].str.count('remove_from_cart')
data['eventlist_basket_open'] = data['eventstring'].str.count('basket_open')
data['eventlist_product_click'] = data['eventstring'].str.count('product_click')
data['eventlist_clear_basket'] = data['eventstring'].str.count('clear_basket')
data['eventlist_add_all_to_cart'] = data['eventstring'].str.count('add_all_to_cart')


PadLength = data['eventlist_len'].max()


event_enum = ['add_to_cart', 'homepage_view', 'product_view', 'list_impression', 'remove_from_cart', 'basket_open', 'product_click', 'clear_basket', 'add_all_to_cart']
def mapper(x):
    for ix, i in enumerate(event_enum):
        if i in x:
            return ix+1
    return 0

## Encode the eventList into a vector of Numbers
data['eventlist_coded'] = data['eventlist'].apply(lambda x: [mapper(i) for i in x])


# In[ ]:


## We need to clean the eventlist items so the program can count distinct
def to_1D(series):
    return pd.Series([x for _list in series for x in _list])


# In[ ]:


fig, ax = plt.subplots(figsize = (14,4))
ax.bar(to_1D(data["eventlist_coded"]).value_counts().index,
        to_1D(data["eventlist_coded"]).value_counts().values)
ax.set_ylabel("Frequency", size = 12)
ax.set_title("Event", size = 14)
plt.xticks(rotation = 90) 
plt.show()


# ### Feature Importances using RandomForestRegressor

# In[ ]:


## Check Feature Importances for each variable
X_data = data[['eventlist_len', 'device', 'custype', 'eventlist_add_to_cart', 'eventlist_product_view', 'eventlist_homepage_view', 'eventlist_remove_from_cart', 'eventlist_basket_open', 'eventlist_product_click', 'eventlist_clear_basket', 'eventlist_add_all_to_cart']]
y_data = data['converted'].to_numpy()


# In[ ]:


## Random Forest and RNN models takes numbers only, requiring us to convert any object column to numbers. 
for i in X_data.columns:
    if X_data[i].dtype == 'object':
        X_data[i] = X_data[i].astype('category').cat.codes +1


# In[ ]:


## Define and train the Random Forest model on the training set. We can then see how well the model utilizng the important features performs on the test set.
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_data, y_data)


# In[ ]:


sorted_idx = rf.feature_importances_.argsort()
plt.barh(X_data.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.show()


# ## Models to test best feature set

# ### All features model

# In[ ]:


LSTM_model_allfeatures = joblib.load('lstm_model_all_2.npy')


# In[ ]:


callback = callbacks.EarlyStopping(monitor="val_accuracy",patience= 5,restore_best_weights=True)
# design network
def model_LSTM(seq_length: int):
    model = Sequential()
    model.add(Embedding(input_dim = 15, output_dim = 100, input_length = seq_length))
    model.add(LSTM(128, activation='tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(50 , activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


# ### All features

# In[ ]:


## Convert the Categorically Encoded EventList to a fixed size Matrix for inputting to the models. This requires getting the value of the longest sequence to use as the length.
## Shorter sequences will have 0's padded on the left
X = data['eventlist_coded'].apply(lambda x: np.pad(x, (PadLength-len(x), 0), 'constant',constant_values=(0, 0)))
X = np.stack(X.values).reshape(-1, PadLength) 
continuous = ["device", "custype"]
trainContinuous = X_data[continuous]
X_all = np.hstack([trainContinuous, X])
y = y_data


# In[ ]:


## Split the Dataset into training and testing sets.
X_train_all, X_rem, y_train, y_rem = train_test_split(X_all, y, random_state=1, test_size=0.3)
X_val_all, X_test_all, y_val, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


# In[ ]:


## LSTM
LSTM_model_allfeatures = KerasClassifier(model=lambda: model_LSTM(X_train_all.shape[1]), verbose=1)
LSTM_pipe = Pipeline([('LSTM_model', LSTM_model_allfeatures)])

# parameters for GridSearch
batchsize = [200, 400]
params = {'LSTM_model__batch_size': batchsize}

# fit model using GridSearch
gs_LSTM_all = GridSearchCV(LSTM_pipe,
                      param_grid=params,
                      scoring='accuracy',
                       refit=False,
                      cv = 5)

gs_LSTM_all.fit(X_train_all, y_train)
# find best hyperparameters
print(gs_LSTM_all.best_params_)


# In[ ]:


pd.DataFrame(gs_LSTM_all.cv_results_).iloc[0,0:4]


# In[ ]:


# Refit the model with the best parameters
LSTM_model_allfeatures = model_LSTM(X_train_all.shape[1])
LSTM_model_allfeatures.fit(X_train_all, y_train, epochs = 30, batch_size = 200, verbose=1, validation_data = (np.asarray(X_val_all), np.asarray(y_val)), callbacks = [callback])
LSTM_predictions_val_all = LSTM_model_allfeatures.predict(np.asarray(X_val_all))
LSTM_predictions_test_all = LSTM_model_allfeatures.predict(np.asarray(X_test_all))


# In[ ]:


#Confusion matrix voor validation
conf_matrix_val = confusion_matrix(y_val, np.round(LSTM_predictions_val_all))
#Confusion matrix voor test
conf_matrix_test = confusion_matrix(y_test, np.round(LSTM_predictions_test_all))


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_val, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_val.shape[0]):
    for j in range(conf_matrix_val.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_val[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_test, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_test.shape[0]):
    for j in range(conf_matrix_test.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_test[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


# predict probabilities for test set
yhat_probs = LSTM_model_allfeatures.predict(X_test_all, verbose=1)
# predict crisp classes for test set
yhat_classes = (LSTM_model_allfeatures.predict(X_test_all) > 0.5).astype("int32")

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# ### Device model

# In[ ]:


callback = callbacks.EarlyStopping(monitor="val_accuracy",patience= 5,restore_best_weights=True)
# design network
def model_LSTM(seq_length: int):
    model = Sequential()
    model.add(Embedding(input_dim = 13, output_dim = 100, input_length = seq_length))
    model.add(LSTM(128, activation='tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(50 , activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


# ### Device 

# In[ ]:


## Convert the Categorically Encoded EventList to a fixed size Matrix for inputting to the models. This requires getting the value of the longest sequence to use as the length.
## Shorter sequences will have 0's padded on the left
X = data['eventlist_coded'].apply(lambda x: np.pad(x, (PadLength-len(x), 0), 'constant',constant_values=(0, 0)))
X = np.stack(X.values).reshape(-1, PadLength) 
continuous = ["device"]
trainContinuous = X_data[continuous]
X_device = np.hstack([trainContinuous, X])
y = y_data


# In[ ]:


## Split the Dataset into training and testing sets.
X_train_device, X_rem, y_train, y_rem = train_test_split(X_device, y, random_state=1, test_size=0.3)
X_val_device, X_test_device, y_val, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


# In[ ]:


## LSTM
LSTM_model_device = KerasClassifier(model=lambda: model_LSTM(X_train_device.shape[1]), verbose=1)
LSTM_pipe = Pipeline([('LSTM_model', LSTM_model_device)])

# parameters for GridSearch
# May have to be changed according to the actual data
batchsize = [200,400]
params = {'LSTM_model__batch_size': batchsize}

# fit model using GridSearch
gs_LSTM_device = GridSearchCV(LSTM_pipe,
                      param_grid=params,
                      scoring='accuracy',
                       refit=False,
                      cv = 5)

gs_LSTM_device.fit(X_train_device, y_train)
# find best hyperparameters
print(gs_LSTM_device.best_params_)


# In[ ]:


pd.DataFrame(gs_LSTM_device.cv_results_).iloc[0,0:4]


# In[ ]:


# refit the model with the best parameters
LSTM_model_device = model_LSTM(X_train_device.shape[1])
LSTM_model_device.fit(X_train_device, y_train, epochs = 30, batch_size = 200, verbose=1, validation_data = (np.asarray(X_val_device), np.asarray(y_val)), callbacks = [callback]) 
LSTM_predictions_val_device = LSTM_model_device.predict(np.asarray(X_val_device))
LSTM_predictions_test_device = LSTM_model_device.predict(np.asarray(X_test_device))


# In[ ]:


#Confusion matrix voor validation
conf_matrix_val = confusion_matrix(y_val, np.round(LSTM_predictions_val_device))
#Confusion matrix voor test
conf_matrix_test = confusion_matrix(y_test, np.round(LSTM_predictions_test_device))


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_val, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_val.shape[0]):
    for j in range(conf_matrix_val.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_val[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_test, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_test.shape[0]):
    for j in range(conf_matrix_test.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_test[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


# predict probabilities for test set
yhat_probs = LSTM_model_device.predict(X_test_device, verbose=1)
# predict crisp classes for test set
yhat_classes = (LSTM_model_device.predict(X_test_device) > 0.5).astype("int32")

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# ## Customer type model

# In[ ]:


callback = callbacks.EarlyStopping(monitor="val_accuracy",patience= 5,restore_best_weights=True)
# design network
def model_LSTM(seq_length: int):
    model = Sequential()
    model.add(Embedding(input_dim = 12, output_dim = 100, input_length = seq_length))
    model.add(LSTM(128, activation='tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


# ### Customer Type

# In[ ]:


## Convert the Categorically Encoded EventList to a fixed size Matrix for inputting to the models. This requires getting the value of the longest sequence to use as the length.
## Shorter sequences will have 0's padded on the left
X = data['eventlist_coded'].apply(lambda x: np.pad(x, (PadLength-len(x), 0), 'constant',constant_values=(0, 0)))
X = np.stack(X.values).reshape(-1, PadLength) 
continuous = ["custype"]
trainContinuous = X_data[continuous]
X_channel = np.hstack([trainContinuous, X])
y = y_data


# In[ ]:


## Split the Dataset into training and testing sets.
X_train_channel, X_rem, y_train, y_rem = train_test_split(X_device, y, random_state=1, test_size=0.3)
X_val_channel, X_test_channel, y_val, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


# In[ ]:


## LSTM
LSTM_model_channel = KerasClassifier(model=lambda: model_LSTM(X_train_channel.shape[1]), verbose=1)
LSTM_pipe = Pipeline([('LSTM_model', LSTM_model_channel)])

# parameters for GridSearch
batchsize = [200, 400]
params = {'LSTM_model__batch_size': batchsize}

# fit model using GridSearch
gs_LSTM_channel = GridSearchCV(LSTM_pipe,
                      param_grid=params,
                      scoring='accuracy',
                       refit=False,
                      cv = 5)

gs_LSTM_channel.fit(X_train_channel, y_train)
# find best hyperparameters
print(gs_LSTM_channel.best_params_)


# In[ ]:


pd.DataFrame(gs_LSTM_channel.cv_results_).iloc[0,0:4]


# In[ ]:


# refit the model with the best parameters
LSTM_model_channel = model_LSTM(X_train_channel.shape[1])
LSTM_model_channel.fit(X_train_channel, y_train, epochs = 30, batch_size = 200, verbose=1, validation_data = (np.asarray(X_val_channel), np.asarray(y_val)), callbacks = [callback])
LSTM_predictions_val_channel = LSTM_model_channel.predict(np.asarray(X_val_channel))
LSTM_predictions_test_channel = LSTM_model_channel.predict(np.asarray(X_test_channel))


# In[ ]:


#Confusion matrix voor validation
conf_matrix_val = confusion_matrix(y_val, np.round(LSTM_predictions_val_channel))
#Confusion matrix voor test
conf_matrix_test = confusion_matrix(y_test, np.round(LSTM_predictions_test_channel))


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_val, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_val.shape[0]):
    for j in range(conf_matrix_val.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_val[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_test, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_test.shape[0]):
    for j in range(conf_matrix_test.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_test[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


yhat_probs = LSTM_model_channel.predict(X_test_channel, verbose=1)
# predict crisp classes for test set
yhat_classes = predictions = (LSTM_model_channel.predict(X_test_channel) > 0.5).astype("int32")
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# ### Sequence model

# In[ ]:


callback = callbacks.EarlyStopping(monitor="val_accuracy",patience= 5,restore_best_weights=True)
def model_LSTM(seq_length: int):
    model = Sequential()
    model.add(Embedding(input_dim = 10, output_dim = 100, input_length = seq_length)) #X_train_LSTM.shape[1]
    model.add(LSTM(128, activation='tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


# ### Sequence only

# In[ ]:


## Convert the Categorically Encoded EventList to a fixed size Matrix for inputting to the models. This requires getting the value of the longest sequence to use as the length.
## Shorter sequences will have 0's padded on the left
X = data['eventlist_coded'].apply(lambda x: np.pad(x, (PadLength-len(x), 0), 'constant',constant_values=(0, 0)))
X = np.stack(X.values).reshape(-1, PadLength) 
y = y_data


# In[ ]:


## Split the Dataset into training and testing sets.
X_train, X_rem, y_train, y_rem = train_test_split(X, y, random_state=1, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


# In[ ]:


## LSTM
LSTM_model = KerasClassifier(model=lambda: model_LSTM(X_train.shape[1]), verbose=1)
LSTM_pipe = Pipeline([('LSTM_model', LSTM_model)])

# parameters for GridSearch
# May have to be changed according to the actual data
batchsize = [200, 400]
params = {'LSTM_model__batch_size': batchsize}

# fit model using GridSearch
gs_LSTM = GridSearchCV(LSTM_pipe,
                      param_grid=params,
                      scoring='accuracy',
                       refit=False,
                      cv = 5)
gs_LSTM.fit(X_train, y_train)

# find best hyperparameters
print(gs_LSTM.best_params_)


# In[ ]:


# Show CrossValidation Results
pd.DataFrame(gs_LSTM.cv_results_).iloc[0,0:4]


# In[ ]:


# refit the model with the best parameters
LSTM_model = model_LSTM(X_train.shape[1])
LSTM_history = LSTM_model.fit(X_train, y_train, epochs = 30, batch_size = 200, verbose=0, validation_data = (np.asarray(X_val), np.asarray(y_val)), callbacks = [callback])


# In[ ]:


LSTM_model = model_LSTM(X_train.shape[1])
LSTM_model.fit(X_train, y_train, epochs = 30, batch_size = 200, verbose=1, validation_data = (np.asarray(X_val), np.asarray(y_val)), callbacks = [callback])
#We save the model in order to use it in real-time
joblib.dump(LSTM_model, 'lstm_model_2.npy')
LSTM_predictions = LSTM_model.predict(np.asarray(X_val))
LSTM_predictions_test_baseline = LSTM_model.predict(np.asarray(X_test))


# In[ ]:


#Check the probability's of one session
SampleIdx = 400
SampleRow = X_test[SampleIdx]
Sample = np.array([np.pad(SampleRow,((i,0)), mode='constant')[:-i if i>0 else SampleRow.shape[0]] for i in range(PadLength)])[::-1]
print(Sample)
SampleOutput = LSTM_model.predict(Sample)
plt.figure(figsize=(20,5))
plt.plot(SampleOutput[-np.trim_zeros(SampleRow).shape[0]:], color='r',marker='o')
plt.title('LSTM Predicted Probability', fontsize = 20)
plt.xlabel('Number of Events', fontsize = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylabel('Probability of Conversion', fontsize = 20)


# In[ ]:


#Review the model loss and accuracy per epoch
fig, ax = plt.subplots(1, 2, figsize=(6,4))

ax[0].plot(range(1, len(LSTM_history.history["loss"]) + 1), LSTM_history.history["loss"])
ax[0].plot(range(1, len(LSTM_history.history["val_loss"]) + 1), LSTM_history.history["val_loss"])
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(['training', 'validation'])

ax[1].plot(range(1, len(LSTM_history.history["accuracy"]) + 1), LSTM_history.history["accuracy"])
ax[1].plot(range(1, len(LSTM_history.history["val_accuracy"]) + 1), LSTM_history.history["val_accuracy"])
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend(['training', 'validation'])

plt.tight_layout()


# In[ ]:


#Confusion matrix voor validation
conf_matrix_val = (confusion_matrix(y_val, np.round(LSTM_predictions)))
#Confusion matrix voor test
conf_matrix_test= confusion_matrix(y_test, np.round(LSTM_predictions_test_baseline))


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_val, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_val.shape[0]):
    for j in range(conf_matrix_val.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_val[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_test, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_test.shape[0]):
    for j in range(conf_matrix_test.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_test[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


yhat_probs = LSTM_model.predict(X_test, verbose=1)
yhat_classes = (LSTM_model.predict(X_test) > 0.5).astype("int32")

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# ## Comparing LSTM to RNN and GRU

# ### RNN

# In[ ]:


callback = callbacks.EarlyStopping(monitor="val_accuracy",patience= 5,restore_best_weights=True)
# design network
def model_RNN(seq_length: int):
    model = Sequential()
    model.add(Embedding(input_dim = 10, output_dim = 100, input_length = seq_length))
    model.add(SimpleRNN(52, activation='tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


# In[ ]:


## RNN
RNN_model = KerasClassifier(model=lambda: model_RNN(X_train.shape[1]), verbose=1)
RNN_pipe = Pipeline([('RNN_model', RNN_model)])

# parameters for GridSearch
# May have to be changed according to the actual data
batchsize = [200, 400]
params = {'RNN_model__batch_size': batchsize}

# fit model using GridSearch
gs_RNN = GridSearchCV(RNN_pipe,
                      param_grid=params,
                      scoring='accuracy',
                       refit=False,
                      cv = 5)
gs_RNN.fit(X_train, y_train, )

# find best hyperparameters
print(gs_RNN.best_params_)


# In[ ]:


# Show CrossValidation Results
pd.DataFrame(gs_RNN.cv_results_).iloc[0,0:4]


# In[ ]:


# refit the model with the best parameters
RNN_model = model_RNN(X_train.shape[1])
RNN_history = RNN_model.fit(X_train, y_train, epochs = 30, batch_size = 400, verbose=0, validation_data = (np.asarray(X_val), np.asarray(y_val)), callbacks = [callback])


# In[ ]:


# refit the model with the best parameters
RNN_model = model_RNN(X_train.shape[1])
RNN_model.fit(X_train, y_train, epochs = 30, batch_size = 400, verbose=1, validation_data = (np.asarray(X_val), np.asarray(y_val)), callbacks = [callback])
#We save the model in order to use it in real-time
joblib.dump(RNN_model, 'rnn_model_2.npy')
RNN_predictions = RNN_model.predict(np.asarray(X_val))
RNN_predictions_test_baseline = RNN_model.predict(np.asarray(X_test))


# In[ ]:


#Check the probability's of one session
SampleIdx = 400
SampleRow = X_test[SampleIdx]
Sample = np.array([np.pad(SampleRow,((i,0)), mode='constant')[:-i if i>0 else SampleRow.shape[0]] for i in range(PadLength)])[::-1]
print(Sample)
SampleOutput = RNN_model.predict(Sample)
plt.figure(figsize=(20,5))
plt.plot(SampleOutput[-np.trim_zeros(SampleRow).shape[0]:], color='r',marker='o')
plt.title('RNN Predicted Probability', fontsize = 20)
plt.xlabel('Number of Events', fontsize = 20)
plt.ylabel('Probability of Conversion', fontsize = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)


# In[ ]:


#Review the model loss and accuracy per epoch
fig, ax = plt.subplots(1, 2, figsize=(6,4))

ax[0].plot(range(1, len(RNN_history.history["loss"]) + 1), RNN_history.history["loss"])
ax[0].plot(range(1, len(RNN_history.history["val_loss"]) + 1), RNN_history.history["val_loss"])
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(['training', 'validation'])

ax[1].plot(range(1, len(RNN_history.history["accuracy"]) + 1), RNN_history.history["accuracy"])
ax[1].plot(range(1, len(RNN_history.history["val_accuracy"]) + 1), RNN_history.history["val_accuracy"])
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend(['training', 'validation'])

plt.tight_layout()


# In[ ]:


#Confusion matrix voor validation
conf_matrix_val = confusion_matrix(y_val, np.round(RNN_predictions))
#Confusion matrix voor test
conf_matrix_test = confusion_matrix(y_test, np.round(RNN_predictions_test_baseline))


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_val, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_val.shape[0]):
    for j in range(conf_matrix_val.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_val[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_test, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_test.shape[0]):
    for j in range(conf_matrix_test.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_test[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


yhat_probs = RNN_model.predict(X_test, verbose=1)
yhat_classes = (RNN_model.predict(X_test) > 0.5).astype("int32")

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# ### GRU

# In[ ]:


callback = callbacks.EarlyStopping(monitor="val_accuracy",patience= 5,restore_best_weights=True)
# design network
def model_GRU(seq_length: int):
    model = Sequential()
    model.add(Embedding(input_dim = 10, output_dim = 100, input_length = seq_length))
    model.add(GRU(128, activation='tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(GRU(50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model


# In[ ]:


## GRU
GRU_model = KerasClassifier(model=lambda: model_GRU(X_train.shape[1]), verbose=1)
GRU_pipe = Pipeline([('GRU_model', GRU_model)])

# parameters for GridSearch
# May have to be changed according to the actual data
batchsize = [200, 400]
params = {'GRU_model__batch_size': batchsize}

# fit model using GridSearch
gs_GRU = GridSearchCV(GRU_pipe,
                      param_grid=params,
                      scoring='accuracy',
                       refit=False,
                      cv = 5)
gs_GRU.fit(X_train, y_train, )

# find best hyperparameters
print(gs_GRU.best_params_)


# In[ ]:


# Show CrossValidation Results
pd.DataFrame(gs_GRU.cv_results_).iloc[0,0:4]


# In[ ]:


GRU_model = model_GRU(X_train.shape[1])
GRU_history = GRU_model.fit(X_train, y_train, epochs=30, batch_size = 200, verbose=0, validation_data = (np.asarray(X_val), np.asarray(y_val)), callbacks = [callback])


# In[ ]:


GRU_model = model_GRU(X_train.shape[1])
GRU_model.fit(X_train, y_train, epochs=30, batch_size = 200, verbose=1, validation_data = (np.asarray(X_val), np.asarray(y_val)), callbacks = [callback])
joblib.dump(GRU_model, 'gru_model_2.npy')
GRU_predictions = GRU_model.predict(np.asarray(X_val))
GRU_predictions_test_baseline = GRU_model.predict(np.asarray(X_test))


# In[ ]:


SampleIdx = 400
SampleRow = X_test[SampleIdx]
Sample = np.array([np.pad(SampleRow,((i,0)), mode='constant')[:-i if i>0 else SampleRow.shape[0]] for i in range(PadLength)])[::-1]
print(Sample)
SampleOutput = GRU_model.predict(Sample)
plt.figure(figsize=(20,5))
plt.plot(SampleOutput[-np.trim_zeros(SampleRow).shape[0]:], color='r',marker='o')
plt.title('GRU Predicted Probability', fontsize = 20)
plt.xlabel('Number of Events', fontsize = 20)
plt.ylabel('Probability of Conversion', fontsize = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(6,4))

ax[0].plot(range(1, len(GRU_history.history["loss"]) + 1), GRU_history.history["loss"])
ax[0].plot(range(1, len(GRU_history.history["val_loss"]) + 1), GRU_history.history["val_loss"])
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(['training', 'validation'])

ax[1].plot(range(1, len(GRU_history.history["accuracy"]) + 1), GRU_history.history["accuracy"])
ax[1].plot(range(1, len(GRU_history.history["val_accuracy"]) + 1), GRU_history.history["val_accuracy"])
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend(['training', 'validation'])

plt.tight_layout()


# In[ ]:


#Confusion matrix voor validation
conf_matrix_val = confusion_matrix(y_val, np.round(GRU_predictions))
#Confusion matrix voor test
conf_matrix_test = confusion_matrix(y_test, np.round(GRU_predictions_test_baseline))


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_val, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_val.shape[0]):
    for j in range(conf_matrix_val.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_val[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix_test, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix_test.shape[0]):
    for j in range(conf_matrix_test.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_test[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


yhat_probs = GRU_model.predict(X_test, verbose=1)
yhat_classes = (GRU_model.predict(X_test) > 0.5).astype("int32")

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# ## Evaluation of Real-Time data

# ### LSTM-RNN

# In[ ]:


lstm_realtime_probs = np.loadtxt('lstm_probs.csv', delimiter=',')
lstm_realtime_probs= lstm_realtime_probs.reshape(29993, 1)
lstm_realtime_classes = np.loadtxt('lstm_classes.csv', delimiter=',')
lstm_realtime_classes= lstm_realtime_classes.reshape(29993, 1)


# In[ ]:


# reduce to 1d array
yhat_probs = lstm_realtime_probs[:, 0]
yhat_classes = lstm_realtime_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_realtime_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_realtime_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_realtime_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_realtime_test, yhat_classes)
print('F1 score: %f' % f1)


# ### RNN

# In[ ]:


rnn_realtime_probs = np.loadtxt('rnn_probs.csv', delimiter=',')
rnn_realtime_probs= rnn_realtime_probs.reshape(29993, 1)
rnn_realtime_classes = np.loadtxt('rnn_classes.csv', delimiter=',')
rnn_realtime_classes= rnn_realtime_classes.reshape(29993, 1)


# In[ ]:


# reduce to 1d array
yhat_probs = rnn_realtime_probs[:, 0]
yhat_classes = rnn_realtime_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_realtime_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_realtime_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_realtime_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_realtime_test, yhat_classes)
print('F1 score: %f' % f1)


# ### GRU

# In[ ]:


gru_realtime_probs = np.loadtxt('gru_probs.csv', delimiter=',')
gru_realtime_probs= gru_realtime_probs.reshape(29993, 1)
gru_realtime_classes = np.loadtxt('gru_classes.csv', delimiter=',')
gru_realtime_classes= gru_realtime_classes.reshape(29993, 1)


# In[ ]:


# reduce to 1d array
yhat_probs = gru_realtime_probs[:, 0]
yhat_classes = gru_realtime_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_realtime_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_realtime_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_realtime_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_realtime_test, yhat_classes)
print('F1 score: %f' % f1)

