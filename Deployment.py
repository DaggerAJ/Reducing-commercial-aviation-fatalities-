#!/usr/bin/env python
# coding: utf-8

# In[1]:


from catboost import CatBoostClassifier


# In[2]:


from flask import Flask, render_template,request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import timeit
from tornado.platform.asyncio import AsyncIOMainLoop
from sklearn.metrics import log_loss
import scipy.signal as signal

import pickle

current = AsyncIOMainLoop(make_current=True) 


# In[3]:


import flask
app = Flask(__name__)


# In[3]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# def noise_removal(noisy_data,Wn):
#     N = 3
#     B, A = signal.butter(N, Wn)
#     return signal.lfilter(B,A, noisy_data,axis=0)


# In[4]:


# def noise_plus_transform(features,df):
#     for i in features:
#         df[i]=noise_removal(df[i],0.1)
#         df[i]=scaler.fit_transform(df[[i]])
#     return df


# In[5]:


# def x_rem_fe(df):

# #   df=df.drop(['experiment'],axis=1)
# #   encoderpl= {'A':0,'B':1,'C':2,'D':3}
# #   y=df['event'].apply(lambda x: encoderpl[x])

#   return df


# In[6]:


# def auto_encoder(model,df):

#     encoder = load_model(model)
#     X=df[df.columns[0:20]]
# #     encoderpl= {'A':0,'B':1,'C':2,'D':3}
# #     y=df['event'].apply(lambda x: encoderpl[x])
# #     X_train, X_test = train_test_split(X, test_size=0.33, random_state=1)
#     X_train_encode = encoder.predict(X)
#   # encode the test data
# #     X_test_encode = encoder.predict(X_test)
# #     x_final = np.vstack((X_train_encode ,X_test_encode))
#     df['eeg_fe_1'] = X_train_encode[:,0]
#     df['eeg_fe_2'] = X_train_encode[:,1]
#     df['eeg_fe_3'] = X_train_encode[:,2]
#     df['eeg_fe_4'] = X_train_encode[:,3]
#     feats= ['eeg_fe_1','eeg_fe_2','eeg_fe_3','eeg_fe_4']
    
#     df=noise_plus_transform(feats,df)
    
#     return df


# In[7]:


# def pipeline(df,autoencoder):
#     df=noise_plus_transform(df.columns[0:23],df)
#     df=auto_encoder(autoencoder,df)
# #     df=x_rem_fe(df)
#     return df


# In[1]:


def state(y):
        
        if (y==0):
            j=('Predicted cognitive state: Baseline or No event')
        elif (y==1):
            j=('Predicted cognitive state: Surprised or Startle')
        elif (y==2):
            j=('Predicted cognitive state: Channelized attention')
        else:
            j=('Predicted cognitive state: Diverted attention')
        return j


# In[9]:


@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        
#         features = ["eeg_fp1","eeg_f7","eeg_f8","eeg_t4","eeg_t6","eeg_t5","eeg_t3","eeg_fp2","eeg_o1","eeg_p3","eeg_pz","eeg_f3","eeg_fz","eeg_f4","eeg_c4","eeg_p4","eeg_poz","eeg_c3","eeg_cz","eeg_o2"]
#         noise_plus_transform(features,df)

        lst = []
        lst2 = []
        to_predict_list = request.form.to_dict()
        lst.append(to_predict_list['input'])
        lst[0] = lst[0].rstrip()
        b = lst[0].split(',')
        for i in b:
            
            lst2.append(float(i))
        x = pd.DataFrame([lst2], columns=['eeg_fp1', 'eeg_f7', 'eeg_f8', 'eeg_t4', 'eeg_t6', 'eeg_t5', 'eeg_t3',
                                      'eeg_fp2', 'eeg_o1', 'eeg_p3', 'eeg_pz', 'eeg_f3', 'eeg_fz', 'eeg_f4',
                                      'eeg_c4', 'eeg_p4', 'eeg_poz', 'eeg_c3', 'eeg_cz', 'eeg_o2', 'ecg', 'r',
                                      'gsr','crew','time','seat'] )
#         df=pipeline(x,'encoder.h5')
        model = joblib.load('cat_boost.pkl')
#         df=df.drop(['experiment'],axis=1)
        y_test_predictions= model.predict_proba(x)
        final_pred = np.argmax(y_test_predictions, axis=1)
        
        states=state(final_pred)
        return states
        



        


# In[ ]:


if __name__ == "__main__":
  app.run(debug=True)


# In[ ]:




