import tensorflow as tf
import tensorflow_datasets as tfds

train_datasets = tfds.load('cifar10',split='train')
valid_datasets = tfds.load('cifar10',split='test')

for data in train_datasets.take(5):
    # print(data['image'].shape)
    # image의 dtype = uint8 이므로, cast
    image = tf.cast(data['image'], tf.float32) / 255.0
    label = data['label']
    print(image)
    print(label)

def preprocessing(data):
    image = tf.cast(data['image'], tf.float32) / 255.0
    label = data['label']
    return image, label

Batch_size = 128
train_data = train_datasets.map(preprocessing).shuffle(1000).batch(Batch_size)
valid_data = valid_datasets.map(preprocessing).batch(Batch_size)
"""tensorflow data sets Load 완료
모델 학습전 preprocess 까지"""
#Sequential API 사용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(32,32,3)), #32개의 필터,커널 사이즈=3,첫문장에 input_shape
    MaxPool2D(2,2),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(10, activation='softmax') #label이 10개 임으로 output layer 10개 설
])
model.summary( )

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
model.fit(train_data,
          validation_data=(valid_data),
          epochs=10)
#Modeling을 간결하게 했기 때문에 정확도가 높게 나오지 않았다.

#Functional API
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

input_ = Input(shape = ((32,32,3)))

x = Conv2D(32, 3, activation='relu')(input_)
x = MaxPool2D(2,2)(x)
x = Conv2D(64, 3, activation='relu')(x)
x = MaxPool2D(2,2)(x)
x =Flatten()(x)
x = Dense(32,activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(input_,x)
model.summary()

"""Sequential / Functional API의 결과값 차이는 거의 없지만
차후에 코드를 수정할때 편리함"""
