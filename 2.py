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

# 代码使用LSTM,以前30天数据为输入数据，后15天为标签进行训练，然后使用后30天来预测15天，
# 接着再使用预测到的15天已经之前的15天共30天的数据再预测15天，现在最好的成绩是2.43


# 读取感染人数,并且处理
#这里只使用了感染人数和迁移人数

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

# 数据预处理，使用迁移人数来处理感染人数，这里认为迁移人数按照迁移比例改变了相对于的地区的感染人数，
# 下面的处理将这种影响体现出来
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
                # 感染人数为当前感染人数减去迁移人数
                infection[from_].loc[id, 'num_infection'] = infection[from_].loc[id, 'num_infection'] - \
                                                            infection[from_].loc[id, 'num_infection'] * k
                # 记录该城市该天总的迁移人数
                infection_num = infection_num + infection[from_].loc[id, 'num_infection'] * k
            row_id = infection[to_][infection[to_].date == t].index.tolist()
            for id in row_id:
                # 被迁移的城市加上相对应的单位迁移人数
                infection[to_].loc[id, 'num_infection'] = infection[to_].loc[id, 'num_infection'] + infection_num / len(
                    row_id)

#提取训练数据
raw_datas = []
for key in infection.keys():
    region_ids = list(infection[key]['region_id'].values)
    region_id_max = max(region_ids)
    for i in range(region_id_max + 1):
        new = infection[key][infection[key]['region_id'] == i]
        new = new.drop(new.columns[[0, 1, 2]], axis=1)
        raw_datas.append(new.values)

train_data = []
train_lbls = []

test_data = []
predict_data = []
predict_data2 = []

for value in raw_datas:
    lbls = value[:, 0]
    x = value[:30]
    y = lbls[30:45]
    z = value[15:45]
    z_ = value[30:45]
    train_data.append(x)#训练数据
    train_lbls.append(y)#label
    predict_data.append(z)#第一次预测数据，后30天
    predict_data2.append(z_)#第二次预测数据，后15天

data = train_data
target = train_lbls

#构建网络并训练
model = Sequential()
model.add(layers.LSTM(64, input_shape=(30, 1)))
model.add(layers.Dense(15))
model.summary()
adam = keras.optimizers.adam(0.002)
model.compile(optimizer=adam,
              loss="mean_squared_logarithmic_error")

# 每200epoch，学习率减半
def lr_scheduler(epoch):
    if epoch % 200 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
    return K.get_value(model.optimizer.lr)


lrate = LearningRateScheduler(lr_scheduler)

history = model.fit(np.array(data), np.array(target), epochs=1000,
                    batch_size=128, validation_split=0.1,
                    verbose=1, callbacks=[lrate], shuffle=True)

#先预测15天的数据
predict1 = model.predict(np.array(predict_data))

# 将第一次预测数据与后15天数据连接组成30天的数据
for index, value in enumerate(predict1):
    tmp = []
    for v in value:
        if v < 0:
            tmp.append(0)
        else:
            tmp.append(v)
    predict_data2[index] = np.array(list(predict_data2[index]) + tmp).reshape(30, 1)
#再预测15天
predict2 = model.predict(np.array(predict_data2))

#得到最终数据
final_predict = []
for index, value in enumerate(zip(predict1, predict2)):
    tmp = []
    for v in value[0]:
        if v < 0:
            tmp.append(0)
        else:
            tmp.append(v)
    for v in value[1]:
        if v < 0:
            tmp.append(0)
        else:
            tmp.append(v)
    final_predict = final_predict + tmp

#处理，写入对应格式的csv文件
region_id = []
date_list = [21200615, 21200616, 21200617, 21200618, 21200619, 21200620, 21200621, 21200622, 21200623, 21200624,
             21200625, 21200626, 21200627, 21200628, 21200629, 21200630, 21200701, 21200702, 21200703, 21200704,
             21200705, 21200706, 21200707, 21200708, 21200709, 21200710, 21200711, 21200712, 21200713, 21200714]
date = []
city_list = []

for key in infection.keys():
    max_region_id = max(list(infection[key]['region_id'].values))
    for i in range(max_region_id + 1):
        date = date + date_list
        for j in range(0, 30):
            region_id.append(i)
            city_list.append(key)

df_tmp = np.transpose(np.vstack((city_list, region_id, date, final_predict)))
df = pd.DataFrame(df_tmp).infer_objects()
df.to_csv('submit_2.csv', header=False, index=False)
