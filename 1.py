import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras import regularizers
import os
import keras
import pandas as pd

import keras.backend as K

import numpy as np

from keras.callbacks import LearningRateScheduler



# 读取感染人数,并且处理

data_A = pd.read_csv('infection_A.csv', header=0, index_col=None, encoding='utf-8')
data_B = pd.read_csv('infection_B.csv', header=0, index_col=None, encoding='utf-8')
data_C = pd.read_csv('infection_C.csv', header=0, index_col=None, encoding='utf-8')
data_D = pd.read_csv('infection_D.csv', header=0, index_col=None, encoding='utf-8')
data_E = pd.read_csv('infection_E.csv', header=0, index_col=None, encoding='utf-8')

infection = {'A': data_A, 'B': data_B, 'C': data_C, 'D': data_D, 'E': data_E}

migration_A = pd.read_csv('migration_A.csv', header=None, index_col=None, encoding='utf-8')
migration_B = pd.read_csv('migration_B.csv', header=None, index_col=None, encoding='utf-8')
migration_C = pd.read_csv('migration_C.csv', header=None, index_col=None, encoding='utf-8')
migration_D = pd.read_csv('migration_D.csv', header=None, index_col=None, encoding='utf-8')
migration_E = pd.read_csv('migration_E.csv', header=None, index_col=None, encoding='utf-8')

migration = {'A': migration_A, 'B': migration_B, 'C': migration_C, 'D': migration_D, 'E': migration_E}

date_ = [21200501, 21200502, 21200503, 21200504, 21200505, 21200506, 21200507, 21200508,
         21200509, 21200510, 21200511, 21200512, 21200513, 21200514, 21200515, 21200516,
         21200517, 21200518, 21200519, 21200520, 21200521, 21200522, 21200523, 21200524,
         21200525, 21200526, 21200527, 21200528, 21200529, 21200530, 21200531, 21200601,
         21200602, 21200603, 21200604, 21200605, 21200606, 21200607, 21200608, 21200609,
         21200610, 21200611, 21200612, 21200613, 21200614]


for t in date_:
    for key in migration.keys():
        migration_now = migration[key][migration[key][0] == t]
        for row in migration_now.iterrows():
            from_ = row[1][1]
            to_ = row[1][2]
            k = float(row[1][3])
            infection_num = 0
            row_id = infection[from_][infection[from_].date == t].index.tolist()
            for id in row_id:
                infection[from_].loc[id, 'num_infection'] = infection[from_].loc[id, 'num_infection'] - \
                                                            infection[from_].loc[id, 'num_infection'] * k
                infection_num = infection_num + infection[from_].loc[id, 'num_infection'] * k
            row_id = infection[to_][infection[to_].date == t].index.tolist()
            for id in row_id:
                infection[to_].loc[id, 'num_infection'] = infection[to_].loc[id, 'num_infection'] + infection_num/len(row_id)



raw_datas=[]
for key in infection.keys():
    region_ids=list(infection[key]['region_id'].values)
    region_id_max=max(region_ids)
    for i in range(region_id_max+1):
        new = infection[key][infection[key]['region_id'] == i]
        new = new.drop(new.columns[[0,1,2]],axis=1)
        raw_datas.append(new.values)


train_data = []
train_lbls = []

test_data = []
predict_data = []

for value in raw_datas:
    lbls = value[:, 0]
    x = value[:15]
    y = lbls[15:45]
    z = value[30:45]
    train_data.append(x)
    train_lbls.append(y)
    predict_data.append(z)

data = train_data
target = train_lbls


model = Sequential()
model.add(layers.LSTM(64, input_shape=(15, 1)))
model.add(layers.Dense(30))
model.summary()
adam = keras.optimizers.adam(0.002)
model.compile(optimizer=adam,
              loss="mean_squared_logarithmic_error")


def lr_scheduler(epoch):
    if epoch % 200 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

lrate = LearningRateScheduler(lr_scheduler)

history = model.fit(np.array(data), np.array(target), epochs=2000,
                    batch_size=72, validation_split=0.1,
                    verbose=1, callbacks=[lrate], shuffle=True)

final_predict = []
predict = model.predict(np.array(predict_data))

for value in predict:
    tmp = []
    for v in value:
        if v < 0:
            tmp.append(0)
        else:
            tmp.append(v)
    final_predict = final_predict + tmp
region_id = []
date_list = [21200615, 21200616, 21200617, 21200618, 21200619, 21200620, 21200621, 21200622, 21200623, 21200624,
             21200625, 21200626, 21200627, 21200628, 21200629, 21200630, 21200701, 21200702, 21200703, 21200704,
             21200705, 21200706, 21200707, 21200708, 21200709, 21200710, 21200711, 21200712, 21200713, 21200714]
date = []
city_list = []

for key in infection.keys():
    max_region_id=max(list(infection[key]['region_id'].values))
    for i in range(max_region_id+1):
        date=date+date_list
        for j in range(0,30):
            region_id.append(i)
            city_list.append(key)


df_tmp = np.transpose(np.vstack((city_list, region_id, date, final_predict)))
df = pd.DataFrame(df_tmp).infer_objects()
df.to_csv('submit.csv', header=False, index=False)
