# cat vs dog 의 이진분류
# 전처리에 유의한다

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

dataset_name = 'cats_vs_dogs'
train_dataset = tfds.load(name = dataset_name, split= 'train[:80%]')
valid_dataset = tfds.load(name = dataset_name, split= 'train[80%:]')
# tensorflow의 dataset에 들어가서 확인해보면, split에 train데이터만 존재한다. 위처러 valid데이터셋을 따로 지정해 줘야한다.

#for data in train_dataset.take(3):

def preprocess(data):
    x = data['image']
    y = data['label']
    #Define x and y data
    x = tf.cast(x, tf.float32)/255.0 #image Normalization
    x = tf.image.resize(x,(224, 224)) # 서로다른 크기를 갖은 이미지들의 사이즈를 맞춰준다.
    # one_hot encoding이 되어있지 않다.
    return x, y

batch_size = 32
train_data = train_dataset.map(preprocess).batch(batch_size)
valid_data = valid_dataset.map(preprocess).batch(batch_size)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPool2D(2,2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Conv2D(256, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    Flatten(),
    Dropout(0.5), #prevent over_fitting
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax'),

])
model.summary()

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics='acc')
checkpoint_path = "my_checkpoint.ckpt"
checkpoint = ModelCheckpoint(
    filepath='my_checkpoint.ckpt',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    verbose = 1
)
model.fit(train_data,
          validation_data=valid_data,
          epochs=20,
          callbacks=[checkpoint]
          )
model.load_weights(checkpoint_path)

