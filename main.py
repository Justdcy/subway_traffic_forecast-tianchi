# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:38:01 2019

@author: dell
"""
###############################################################
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import norm, rankdata
import warnings
import gc
import os
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

plt.style.use('seaborn')
sns.set(font_scale=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV

path  = '/Users/dell/Desktop/tianchi_AI'
test = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
test_28 = pd.read_csv(path + '/Metro_testA/testA_record_2019-01-28.csv')
station_con = pd.read_csv('Metro_roadMap.csv')

#计算每个站口相连地铁站口个数,后面可选择用或不用，我们用了没提分，或许打开方式不对
del station_con['Unnamed: 0']
station_con_sum=pd.DataFrame()
station_con_sum['station_con_sum'] = np.sum(station_con,axis=0)
station_con_sum = station_con_sum[0:]
station_con_sum['stationID'] = np.arange(81)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def get_base_features(df_,test,time_str):
    
    df = df_.copy()
    #df1 = pd.get_dummies(df,columns=['lineID'])
    df['startTime'] = df['time'].apply(lambda x: x[:15].replace(time_str, '-01-29') + '0:00')

    
    df = df.groupby(['startTime','stationID']).status.agg(['count', 'sum']).reset_index()
    df = test.merge(df, 'left', ['stationID','startTime'])


    df['time'] = df['startTime'].apply(lambda x: x[:15].replace('-01-29', time_str) + '0:00')

    del df['startTime'],df['endTime']
    
    # base time
    df['day']     = df['time'].apply(lambda x: int(x[8:10]))
    
    df['week']    = pd.to_datetime(df['time']).dt.dayofweek + 1
    #df['weekend'] = (pd.to_datetime(df.time).dt.weekday >=5).astype(int)
    df['hour']    = df['time'].apply(lambda x: int(x[11:13]))
    df['minute']  = df['time'].apply(lambda x: int(x[14:15]+'0'))    

    result = df.copy()    
  
    # in,out
    result['inNums']  = result['sum']
    result['outNums'] = result['count'] - result['sum']
    #
    result['day_since_first'] = result['day'] - 1 
    
    ###rank复赛记得有提分，初赛没有用起来，当时打开方式不对
    #result['rank'] = (result['stationID']+1)*(result['day']*144+result['hour']*6+result['minute'])
    result.fillna(0, inplace=True)
    del result['sum'],result['count']
    
    return result


time_str = '-01-28'
data1 = get_base_features(test_28,test,time_str)

###29号时间等信息是本身的，inNums和outNums用的28号的数据
###后面也就可以直接将29号作为测试集
time_str = '-01-29'
df = pd.read_csv(path + '/Metro_testA/testA_record_2019-01-28.csv')

df['time'] = df['time'].apply(lambda x: x[:15].replace('-01-28', time_str)+ '0:00')
df = get_base_features(df,test,time_str)
data1 = pd.concat([data1, df], axis=0, ignore_index=True)


data_list = os.listdir(path+'/Metro_train/')
for i in range(0, len(data_list)):
    if data_list[i].split('.')[-1] == 'csv':
        time_str = data_list[i].split('.')[0][11:17]
        print(data_list[i], i)
        df = pd.read_csv(path+'/Metro_train/' + data_list[i])
        df = get_base_features(df,test,time_str)
        data1 = pd.concat([data1, df], axis=0, ignore_index=True)
    else:
        continue
    
###merge每个站口相连地铁站口个数
#data1 = data1.merge(station_con_sum, on=['stationID'], how='left')

###特征自己可以添加，
def more_feature(result):
    tmp = result.copy()
    tmp = tmp[['stationID','week','day','hour']]
    ###按week计算每个站口每小时客流量特征
    tmp = result.groupby(['stationID','week','hour'], as_index=False)['inNums'].agg({
                                                                        'inNums_ID_dh_max'    : 'max',###
                                                                        'inNums_ID_dh_min'    : 'min',###
                                                                        'inNums_ID_dh_mean'   : 'mean',###
                                                                        'inNums_ID_dh_sum'   : 'sum'
                                                                        })
    result = result.merge(tmp, on=['stationID','week','hour'], how='left')
    ###按week计算每个站口客流量特征
    tmp = result.groupby(['stationID','week'], as_index=False)['inNums'].agg({
                                                                        'inNums_ID_d_max'    : 'max',
                                                                        'inNums_ID_d_min'    : 'min', #都为0
                                                                        'inNums_ID_d_mean'   : 'mean',##
                                                                        'inNums_ID_d_sum'   : 'sum'
                                                                        })
    result = result.merge(tmp, on=['stationID','week'], how='left')
    
    ###每个站口所有天客流量特征
    tmp = result.groupby(['stationID'], as_index=False)['inNums'].agg({
                                                                        'inNums_ID_max'    : 'max',
                                                                        'inNums_ID_min'    : 'min',
                                                                        'inNums_ID_mean'   : 'mean',##
                                                                        'inNums_ID_sum'   : 'sum'
                                                                        })
    result = result.merge(tmp, on=['stationID'], how='left')
    ###每天所有站口客流量特征
    tmp = result.groupby(['day'], as_index=False)['inNums'].agg({
                                                                        'inNums_d_max'    : 'max',
                                                                        'inNums_d_min'    : 'min',#都为0
                                                                        'inNums_d_mean'   : 'mean',##
                                                                        'inNums_d_sum'   : 'sum'
                                                                        })
    result = result.merge(tmp, on=['day'], how='left')
    
    
    
    ###出站与进站类似
    tmp = result.groupby(['stationID','week','hour'], as_index=False)['outNums'].agg({
                                                                        'outNums_ID_dh_max'    : 'max',
                                                                        'outNums_ID_dh_min'    : 'min',##
                                                                        'outNums_ID_dh_mean'   : 'mean',##
                                                                        'outNums_ID_dh_sum'   : 'sum'
                                                                        })
    result = result.merge(tmp, on=['stationID','week','hour'], how='left')
    
    tmp = result.groupby(['stationID','week'], as_index=False)['outNums'].agg({
                                                                        'outNums_ID_d_max'    : 'max',
                                                                        'outNums_ID_d_min'    : 'min',#都为0
                                                                        'outNums_ID_d_mean'   : 'mean',##
                                                                        'outNums_ID_d_sum'   : 'sum'
                                                                        })
    result = result.merge(tmp, on=['stationID','week'], how='left')
    
    
    tmp = result.groupby(['stationID'], as_index=False)['outNums'].agg({
                                                                        'outNums_ID_max'    : 'max',
                                                                        'outNums_ID_min'    : 'min',
                                                                        'outNums_ID_mean'   : 'mean',
                                                                        'outNums_ID_sum'   : 'sum'
                                                                        })
    result = result.merge(tmp, on=['stationID'], how='left')
    
    tmp = result.groupby(['day'], as_index=False)['outNums'].agg({
                                                                        'outNums_d_max'    : 'max',
                                                                        'outNumss_d_min'    : 'min',#都为0
                                                                        'outNums_d_mean'   : 'mean',
                                                                        'outNums_d_sum'   : 'sum'
                                                                        })
    result = result.merge(tmp, on=['day'], how='left')
    
    return result
    
data2 = more_feature(data1)

# 删除某一类别占比超过90%的列
good_cols = list(data2.columns)
for col in data2.columns:
    rate = data2[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.90:
        good_cols.remove(col)
        print(col,rate)

data2 = data2[good_cols]

'''
###皮尔相关系数
fea_train = data2.copy()
del fea_train['time']
fea_y = fea_train['inNums']
del fea_train['inNums']
del fea_train['outNums']
fe = pd.concat([fea_train, fea_y], axis = 1,ignore_index=False)  
colormap = plt.cm.viridis  
plt.figure(figsize=(30,30))  
plt.title('Pearson Correlation of Features', y=1.05, size=15)  
sns.heatmap(fe.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)  



####xgb特征选择
data_mod = data2.copy()
del data_mod['time']
train = data_mod[data_mod.day<25]

valid = data_mod[data_mod.day==25]

#test = data_in_shfit_temp[data_in_shfit_temp.day==j]

from xgboost import plot_importance
y_train = train['inNums']
y_valid = valid['inNums']
#y_data  = X_data['inNums']

del train['inNums'],valid['inNums']#,X_data['inNums']
del train['outNums'],valid['outNums']#,X_data['outNums']

####xgb的特征选择（不太成功）  
xgb_params = {'eta': 0.004, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,   
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1, 'nthread': 4,'lambda': 1,}  
#X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.15, random_state=5)  
dtrain = xgb.DMatrix(train, y_train)  
dtest = xgb.DMatrix(valid,y_valid)  
num_rounds = 10000  
watchlist = [ (dtrain,'train'), (dtest, 'test') ]  
clf = xgb.train(dtrain=dtrain, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)  
plt.figure(figsize=(35,35))  
plot_importance(clf)  
plt.show()  
'''

###时间shift

##将28，29拼接到最后，整体有序了
data_28 = data2[data2.day==28]
data_29 = data2[data2.day==29]
data2 = data2[(data2.day!=28)&(data2.day!=29)]
data2 = pd.concat([data2, data_28,data_29], axis=0, ignore_index=True)
data = data2.copy()


'''
###移动平均，效果不好
data['inNums'] = data['inNums'].rolling(window=2).mean()
data['outNums'] = data['outNums'].rolling(window=2).mean()
data = data.fillna(0)
data['inNums'] = np.round(data['inNums'])
data['outNums'] = np.round(data['outNums']) 
'''

# 剔除周末
data = data[(data.day!=5)&(data.day!=6)&(data.day!=1)]
data = data[(data.day!=12)&(data.day!=13)]
data = data[(data.day!=19)&(data.day!=20)]
data = data[(data.day!=26)&(data.day!=27)]

###shift时间，144个时间点是一天，选取的近三天的时间及其组合特征
def time_shift(data_in_sta,data_in_shfit_cols,data_out_shfit_cols):
    lag_start=144
    lag_end=144*3
    data_out_sta = data_in_sta.copy()
    for i in range(lag_start, lag_end+1,144):
        for col in data_in_shfit_cols:
            data_in_sta[col+"_lag_{}".format(i)] = data_in_sta[col].shift(i)
            if (col != 'inNums') & (col != 'outNums') &(i==lag_end):
                del data_in_sta[col]
        for col1 in data_out_shfit_cols:
            data_out_sta[col1+"_lag_{}".format(i)] = data_out_sta[col1].shift(i)
            if (col1 != 'inNums') & (col1 != 'outNums') &(i==lag_end):
                del data_out_sta[col1]
        
    return data_in_sta,data_out_sta

###由于只shift inNums和outNums，则先排除其余特征
data_in_shfit = pd.DataFrame()
data_out_shfit = pd.DataFrame()

data_in_shfit_cols = list(data)
data_in_shfit_cols.remove('stationID')
data_in_shfit_cols.remove('time')
data_in_shfit_cols.remove('day')
data_in_shfit_cols.remove('week')
#data_in_shfit_cols.remove('weekend')
data_in_shfit_cols.remove('hour')
data_in_shfit_cols.remove('minute')
data_in_shfit_cols.remove('day_since_first')
#data_in_shfit = data_in_shfit[data_in_shfit_cols]

data_out_shfit_cols = list(data)
data_out_shfit_cols.remove('stationID')
data_out_shfit_cols.remove('time')
data_out_shfit_cols.remove('day')
data_out_shfit_cols.remove('week')
#data_in_shfit_cols.remove('weekend')
data_out_shfit_cols.remove('hour')
data_out_shfit_cols.remove('minute')
data_out_shfit_cols.remove('day_since_first')
#data_out_shfit = data_out_shfit[data_out_shfit_cols]


###对每个站口进行shift操作
for i in range(81):
    data_temp = data[data['stationID'] == i]
    data_in_sta,data_out_sta = time_shift(data_temp,data_in_shfit_cols,data_out_shfit_cols)
    data_in_shfit = pd.concat([data_in_shfit, data_in_sta], axis=0, ignore_index=True)
    data_out_shfit = pd.concat([data_out_shfit, data_out_sta], axis=0, ignore_index=True)


###############################################
###############################################inNums

data_in_shfit_temp = data_in_shfit.copy()

del data_in_shfit_temp['time']
data_in_shfit_temp.fillna(0, inplace=True)

###进行时间序列的交叉验证，比如选取23号作为验证集，则23号前的作为训练集，24号作为测试集，依次类推
###自我感觉比较可靠，基本上线下提分，线上也提分
###特别要注意防止数据泄露
test_day = [23,24,25,28]
error_in = []
for i in test_day:

    if ( (i != 28)&(i!=25) ):
        test = data_in_shfit_temp[data_in_shfit_temp.day==i+1]
        y_test = test['inNums']
        del test['inNums']
        del test['outNums']

    if i==25:
        test = data_in_shfit_temp[data_in_shfit_temp.day==i+3]
        y_test = test['inNums']
        del test['inNums']
        del test['outNums']

    print('###############################inNums验证集',i)

    train = data_in_shfit_temp[data_in_shfit_temp.day<i]
    valid = data_in_shfit_temp[data_in_shfit_temp.day==i]
    y_train = train['inNums']
    y_valid = valid['inNums']
    
    del train['inNums'],valid['inNums']
    del train['outNums'],valid['outNums']

    
    from xgboost import XGBRegressor
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LassoCV, RidgeCV
   
    dtrain = xgb.DMatrix(train, label = y_train)
    dtest = xgb.DMatrix(test)
    dval = xgb.DMatrix(valid, label = y_valid)
    watchlist = [(dtrain, 'train'),(dval, 'val')]
    xgb_params = {'eta': 0.004, 'max_depth': 11, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 4}
    clf = xgb.train(dtrain=dtrain, num_boost_round=10000, evals=watchlist, early_stopping_rounds=100, verbose_eval=100, params=xgb_params)

    if i!=28:
        
        prediction_in = clf.predict(dtest, ntree_limit=clf.best_iteration)
        error = mean_absolute_percentage_error(np.abs(np.round(prediction_in)),y_test)
        error_in.append(error)
        print('验证集：',i)
        print('验证集下一天作为测试集的误差为：',error)
        
print('inNums的CV验证分数：',np.mean(error_in))


###最终预测29号时要加上28号，一共的数据集
X_data= data_in_shfit_temp[data_in_shfit_temp.day<29]
test= data_in_shfit_temp[data_in_shfit_temp.day==29]
valid = data_in_shfit_temp[data_in_shfit_temp.day==28]
y_valid = valid['inNums']
del valid['outNums'],valid['inNums']

y_data  = X_data['inNums']
y_test = test['inNums']
del X_data['inNums'],test['inNums']
del X_data['outNums'],test['outNums']

### all_data
dtrain = xgb.DMatrix(X_data, label = y_data)
dtest = xgb.DMatrix(test)
dval = xgb.DMatrix(valid, label = y_valid)
watchlist = [(dtrain, 'train')]

clf = xgb.train(dtrain=dtrain, num_boost_round=clf.best_iteration, early_stopping_rounds=100, evals=watchlist, verbose_eval=100, params=xgb_params)

prediction_in = clf.predict(dtest, ntree_limit=clf.best_iteration)
prediction = pd.DataFrame()
prediction['inNums'] = prediction_in

prediction['inNums'] = abs(np.round(prediction['inNums']))
error_test_in = mean_absolute_percentage_error(abs(np.round(prediction['inNums'])),y_test)


#############################################################outNums
#############################################################
data_out_shfit_temp = data_out_shfit.copy()

del data_out_shfit_temp['time']
data_out_shfit_temp.fillna(0, inplace=True)


from sklearn.preprocessing import StandardScaler

test_day = [23,24,25,28]
error_out = []
for i in test_day:
    
    if ( (i != 28)&(i!=25) ):
        test_out = data_out_shfit_temp[data_out_shfit_temp.day==i+1]
        y_test_out = test_out['outNums']
        del test_out['inNums']
        del test_out['outNums']
        
    if i==25:
        test_out = data_out_shfit_temp[data_out_shfit_temp.day==i+3]
        y_test_out = test_out['outNums']
        del test_out['inNums']
        del test_out['outNums']
    

    print('###############################outNums验证集',i)
    train_out = data_out_shfit_temp[data_out_shfit_temp.day<i]
    
    valid_out = data_out_shfit_temp[data_out_shfit_temp.day==i]
    
    y_train_out = train_out['outNums']
    y_valid_out = valid_out['outNums']
    

    del train_out['inNums'],valid_out['inNums']
    del train_out['outNums'],valid_out['outNums']
    
    dtrain = xgb.DMatrix(train_out, label = y_train_out)
    dtest = xgb.DMatrix(test_out)
    dval = xgb.DMatrix(valid_out, label = y_valid_out)
    watchlist = [(dtrain, 'train'),(dval, 'val')]
    xgb_params = {'eta': 0.004, 'max_depth': 11, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 4}
    clf = xgb.train(dtrain=dtrain, num_boost_round=10000, evals=watchlist, early_stopping_rounds=100, verbose_eval=100, params=xgb_params)

    if i!=28:
        
        prediction_out = clf.predict(dtest, ntree_limit=clf.best_iteration)
        error = mean_absolute_percentage_error(np.abs(np.round(prediction_out)),y_test_out)
        error_out.append(error)
        print('验证集：',i)
        print('验证集下一天作为测试集的误差为：',error)
        
print('outNums的CV验证分数：',np.mean(error_out))

###最终预测29号时要加上28号，一共的数据集
X_data_out= data_out_shfit_temp[data_out_shfit_temp.day<29]
test_out= data_out_shfit_temp[data_out_shfit_temp.day==29]
valid_out = data_out_shfit_temp[data_out_shfit_temp.day==28]
y_valid_out = valid_out['outNums']
    

del valid_out['inNums'],valid_out['outNums']
y_data_out  = X_data_out['outNums']
y_test_out = test_out['outNums']
del X_data_out['inNums'],test_out['inNums']
del X_data_out['outNums'],test_out['outNums']

### all_data
dtrain = xgb.DMatrix(X_data_out, label = y_data_out)
dtest = xgb.DMatrix(test_out)
dval = xgb.DMatrix(valid, label = y_valid)
watchlist = [(dtrain, 'train')]
clf = xgb.train(dtrain=dtrain, num_boost_round=clf.best_iteration, early_stopping_rounds=100,evals=watchlist, verbose_eval=100, params=xgb_params)

prediction_out = clf.predict(dtest, ntree_limit=clf.best_iteration)
prediction['outNums'] = prediction_out

prediction['outNums'] = abs(np.round(prediction['outNums']))
error_test_out = mean_absolute_percentage_error(abs(np.round(prediction['outNums'])),y_test_out)



print('最终inNums和outNums得分：',(np.mean(error_in)+np.mean(error_out))/2)


'''
sub = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
sub['inNums']   = prediction['inNums'].values
sub['outNums']  = prediction['outNums'].values
# 结果修正
#sub.loc[sub.inNums<0 , 'inNums']  = 0
#sub.loc[sub.outNums<0, 'outNums'] = 0


sub[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('sub1.csv', index=False)


'''
