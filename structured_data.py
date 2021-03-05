import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')

def preprocess(data):
    x = data['features']
    y = data['label']
    y = tf.one_hot(y, 3) #3은 class의 개수를 나타낸다. tensorflow 자료에 나와있음
    return x, y

batch_size = 10
train_data = train_dataset.map(preprocess).batch(batch_size)
valid_data = valid_dataset.map(preprocess).batch(batch_size)

model = tf.keras.models.Sequential([
    Dense(512, activation='relu', input_shape = (4,)), #features 의 개수가 4
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax'),

])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics='acc') #one-hot encoding된 데이터
checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(filepath= checkpoint_path,
                             save_weights_only=True,
                             save_best_only= True,
                             monitor='val_loss',
                             verbose = 1
                             )
history = model.fit(train_data,
                    validation_data=(valid_data),
                    epochs=20,
                    callbacks=[checkpoint])
model.load_weights(checkpoint_path)


# 오차에 대한 시각
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, 21), history.history['loss'])
plt.plot(np.arange(1, 21), history.history['val_loss'])
plt.title('Loss / Val Loss of structured', fontsize = 20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss','val_loss'], fontsize=15)
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, 21), history.history['acc'])
plt.plot(np.arange(1, 21), history.history['val_acc'])
plt.title('Acc / Val Acc', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc', 'val_acc'], fontsize=15)
plt.show()