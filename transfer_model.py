'''전이학습: 이미 학습된 모델을 활용한다. 논문을 참조해 같은 코드로 만들더라도, 학습한 데이터의 수가 달라서 성능차이가 난다.
기존에 학습한 가중치를 가지고 온다는 방법이다.
이번 학습에서는 1000개의 클래스를 갖은 VGG16을 사용할 것이다.
개와 고양이 2개의 클래스만 사용하는 튜닝을 해준다.'''

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16

dataset_name = 'cats_vs_dogs'
train_dataset = tfds.load(name = dataset_name, split= 'train[:80%]')
valid_dataset = tfds.load(name = dataset_name, split= 'train[80%:]')

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


transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# 'imagenet' -> 가중치 가지고오기
# include_top=False -> 튜닝을 가능하게하는 옵션
transfer_model.trainable = False # 가중치를 고정시켜, 학습이 발생하지 않도록 한다(freeze)

model = Sequential([
    transfer_model,
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax') # cat and dog binary classification
])

model.summary() #맨 마지막 필터의 개수를 논문에서 비교해 봐서 잘 가지고 왔는지 확인!

#나머지는 CNN_B.py 의 부분과 동일하다.
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
          epochs=10,
          callbacks=[checkpoint]
          )
model.load_weights(checkpoint_path)

