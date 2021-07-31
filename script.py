#!/usr/bin/env python
# coding: utf-8

# This script contains only feature engineering and model training from which we obtain the best score.

# In[ ]:


#! pip install xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import re

plt.style.use('seaborn')
pd.set_option('display.max_column',100)


# In[ ]:


TrainData = pd.read_csv('train.csv')
TestData = pd.read_csv('test.csv')
SessData = pd.read_csv('sessions.csv')
SessData.rename(columns={'user_id':'id'},inplace = True)


# In[ ]:


action_type = pd.pivot_table(SessData, index = ['id'],columns = ['action_type'],values = 'secs_elapsed',aggfunc=len,fill_value=0).reset_index()
action = pd.pivot_table(SessData, index = ['id'],columns = ['action'],values = 'secs_elapsed',aggfunc=len,fill_value=0).reset_index()
action_detail = pd.pivot_table(SessData, index = ['id'],columns = ['action_detail'],values = 'secs_elapsed',aggfunc=len,fill_value=0).reset_index()
device_type = pd.pivot_table(SessData, index = ['id'],columns = ['device_type'],values = 'secs_elapsed',aggfunc=len,fill_value=0).reset_index()


# In[ ]:


action_type['booking_response'] = action_type['booking_response'].apply(lambda x: 1 if x>0 else 0)
action_type['-unknown-'] = action_type['-unknown-'].apply(lambda x: 1 if x>0 else 0)

action_detail['pending'] = action_detail['pending'].apply(lambda x: 1 if x>0 else 0)
action_detail['at_checkpoint'] = action_detail['at_checkpoint'].apply(lambda x: 1 if x>0 else 0)


# In[ ]:


DataCom = pd.concat((TrainData.drop('country_destination',axis=1),TestData),axis=0,ignore_index=True)
def set_age_group(x):
    if x < 40:
        return 'Young'
    elif x >=40 and x < 60:
        return 'Middle'
    elif x >= 60 and x <= 125:
        return 'Old'
    else:
        return 'Unknown'


def set_age(x):
    if x>=16 and x<=120:
        return x
    elif x<16:
        return np.nan
    elif x>120:
        if 2015-x>16 and 2015-x<110:
            return 2015-x
        else:
            return np.nan

def set_categorial_values(x):
    l = ['first_browser','affiliate_provider','first_device_type','affiliate_channel','first_affiliate_tracked','signup_app']
    thresold = [0.00]*10
    
    i = l.index(x)
    l1 = DataCom[x].value_counts(normalize=True)
    l2 = l1[l1>thresold[i]].index.tolist()
    return DataCom[x].apply(lambda x: x if x in l2 else 'diff')

def feature_engineering(df):
    df['DAC_year'] = np.vstack(df['date_account_created'].astype(str).apply(lambda x: list(map(int, x.split('-')))).values)[:,0]
    df['DAC_month'] = np.vstack(df['date_account_created'].astype(str).apply(lambda x: list(map(int, x.split('-')))).values)[:,1]
    df['DAC_day'] = np.vstack(df['date_account_created'].astype(str).apply(lambda x: list(map(int, x.split('-')))).values)[:,2]
    df['DAC_dayofweek'] = pd.to_datetime(df['date_account_created']).dt.dayofweek
    
    df['TFA_year'] = np.vstack(df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)[:,0]
    df['TFA_month'] = np.vstack(df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)[:,1]
    df['TFA_day'] = np.vstack(df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)[:,2]
    df['TFA_hour'] = np.vstack(df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)[:,3]
    
    df['DFB_year'] = np.vstack(df['date_first_booking'].fillna(-1).astype(str).apply(lambda x: -1 if x=='-1' else list(map(int,x.split('-')))[0] ))
    df['DFB_month'] = np.vstack(df['date_first_booking'].fillna(-1).astype(str).apply(lambda x: -1 if x=='-1' else list(map(int,x.split('-')))[1] ))
    df['DFB_day'] = np.vstack(df['date_first_booking'].fillna(-1).astype(str).apply(lambda x: -1 if x=='-1' else list(map(int,x.split('-')))[2] ))
    df['DFB_dayofweek'] = pd.to_datetime(df['date_first_booking']).dt.dayofweek
    
    df['lag'] = (pd.to_datetime(df.date_account_created)-pd.to_datetime(df.timestamp_first_active,format='%Y%m%d%H%M%S')).dt.days
    
    df['age'] = df['age'].apply(set_age)
    df['age_group'] = df['age'].apply(set_age_group)
    df['age'] = df['age'].fillna(-1)
    
    
    df['has_booked'] = df['date_first_booking'].fillna(-1).apply(lambda x: 0 if x==-1 else 1)
    df['first_affiliate_tracked'] = df['first_affiliate_tracked'].fillna('unknown')
    
    l = ['first_browser','affiliate_provider','first_device_type','affiliate_channel','first_affiliate_tracked','signup_app']
    for x in l:
        df[x] = set_categorial_values(x)
            
    ohe = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 
           'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser','age_group']
    
    for x in ohe:
        combined_data,_ = pd.factorize(df[x],sort=True)
        combined_data = pd.Series(combined_data).astype('int32')
        df[x] = combined_data.values  
    
    droplist = ['date_account_created','timestamp_first_active','date_first_booking','signup_method']
    df = df.drop(droplist,axis=1)
    
    df = pd.merge(df, action_type[['id','booking_response','-unknown-']], how='left', on='id')
    df = pd.merge(df, action[['id','requested', 'confirm_email', 'update', 'cancellation_policies']], how='left', on='id')
    df = pd.merge(df, action_detail[['id','pending', 'at_checkpoint']], how='left', on='id')
    
    df = df.set_index('id')
    
    return df


DataCom = feature_engineering(DataCom)


# In[ ]:


y = TrainData['country_destination']             
labels = y.values
le = LabelEncoder()
y = pd.Series(le.fit_transform(labels))      

X = DataCom[:TrainData.shape[0]]                        # encoded train data x               
df_test = DataCom[TrainData.shape[0]:]                  # encoded test data

x_train, x_test, y_train, y_test = train_test_split(X, y,random_state=31, train_size=0.80,  stratify=y)


# In[ ]:


xgb3 = xgb.XGBClassifier(max_depth=4, learning_rate=0.01, n_estimators=75,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=40,reg_lambda=0.5)
xgb3.fit(X, y)


# In[ ]:


prob = xgb3.predict_proba(df_test)
id_test = TestData['id']

id_final_sub = []  
pred_final_sub = [] 
for i in range(len(id_test)):
    idx = id_test[i]
    id_final_sub += [idx] * 3
    pred_final_sub += le.inverse_transform(np.argsort(prob[i])[::-1])[:3].tolist()

sub = pd.DataFrame(np.column_stack((id_final_sub,pred_final_sub)), columns=['id', 'country_destination'])
sub.to_csv('submission.csv',index=False)

