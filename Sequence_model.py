import csv
import tensorflow as tf
import numpy as np
import urllib
import matplotlib.pyplot as plt
'''csv: comma separated value'''
from tensorflow.keras.layers import LSTM,Lambda, Dense, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber

url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
urllib.request.urlretrieve(url, 'sunspots.csv')

sunspots = []
time_step = []

with open('sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    #delimiter ='.'으로 설정해서 월소가 2개인 리스트로 만들어져 (list index out of range)에러가 발생
    next(reader) #첫줄은 header 파일이기 때문에 skip하는 명령어

    for row in reader:

        sunspots.append(float(row[2])) #3번째 원소에 흑점활동
        time_step.append(int(row[0])) # 1번째 원소에 시간의 흐름
        '''파일의 자료는 문자열로 읽히기 때문에 type casting 해주어야 한다.'''
# print(sunspots[:5])
# print(time_step[:5])
series = np.array(sunspots) #shape = (3235,)
time = np.array(time_step) # list 형태로는 x y 데이터로 입력이 되지 않는다.

plt.figure(figsize=(12,9))
plt.plot(time, series)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()
'''현재 패턴을 기반으로 다른 패턴을 예측하는 작업을 할 것이다'''

# Train, Validation data setting
split_data = 3000

series_train = series[:split_data]
series_valid = series[split_data:]

time_train = time[:split_data]
time_valid = time[split_data:]


window_size = 30
shuffle_size = 1000
batch_size = 32

def window_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis= -1) # 1차원 -> 2차원
    ds = tf.data.Dataset.from_tensor_slices(series) # 데이타셋으로 변환
    ds = ds.window(window_size + 1, shift = 1, drop_remainder = True) # 1씩 이동, 꼬투리 제거
    ds = ds.flat_map(lambda w : w.batch(window_size +1))# x_data 뿐만아니라 y_data까지  포함한 범위로 잘라주기 위해
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1],w[1:])) #각각 x ,y 값의 범위를 지정
    return ds.batch(batch_size).prefetch(1) # 병렬적으로 학습해서 학습속도가 향상된다. 1이 적당함

train_set = window_dataset(series_train,
                           window_size =window_size,
                           batch_size = batch_size,
                           shuffle_buffer=shuffle_size)
valid_set = window_dataset(series_valid,
                           window_size=window_size,
                           batch_size=batch_size,
                           shuffle_buffer=shuffle_size)

model = Sequential([
    tf.keras.layers.Conv1D(60, kernel_size=5,
                           padding='causal',
                           activation='relu',
                           input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True), #겹쳐서 쌓는다
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x : x * 400)

])
model.summary()


loss = Huber()
optimizer = SGD(lr = 1e-5,momentum=0.9)

model.compile(loss = loss, optimizer=optimizer,metrics=['mae']) #평균제곱 오차를 사용

model.compile(optimizer='adam', loss= loss, metrics='acc')

checkpoint_path = "my_checkpoint.ckpt"
checkpoint = ModelCheckpoint(
    filepath='my_checkpoint.ckpt',
    save_weights_only=True,
    save_best_only=True,
    monitor='mae_loss',
    verbose = 1
)
epochs = 100
checkpoint_path = 'tmp_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_mae',
                             verbose=1)
history =model.fit(train_set,
                   validation_data=valid_set,
                   epochs=100,
                   callbacks=[checkpoint])
model.load_weights(checkpoint_path)

#시각화
plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, epochs+1), history.history['loss'])
plt.plot(np.arange(1, epochs+1), history.history['val_loss'])
plt.title('Loss / Val Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'], fontsize=15)
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, epochs+1), history.history['mae'])
plt.plot(np.arange(1, epochs+1), history.history['val_mae'])
plt.title('MAE / Val MAE', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend(['mae', 'val_mae'], fontsize=15)
plt.show()

'''val_mae값이 변동이 심하고 13 이하면 유의미한 값으로 여겨진다
val_mae 값이 25에 35 사이로 큰 변동으로 나와서 문제를 찾아보니 62번째 줄에서 y값의 범위를 지정해 줄때, 범위를 잘못 설정해 주었다.
 (w[:-1],w[-1:])  -> (w[:-1],w[1:])으로 변경했다.'''