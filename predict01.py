import tensorflow as tf
import numpy as np
from tensorflow import keras
#첫번째 과제
#7개의 방의 가격 예측

def house_model(y_new):

    xs = np.arange(1, 11, dtype = float)
    start=1
    step=0.5
    num=10
    ys = np.arange(0,num)*step+start
    model = tf.keras.Sequential(keras.layers.Dense(units = 1, input_shape = [1]))
    model.compile(optimizer = 'sgd', loss= 'mean_squared_error')
    model.fit(xs, ys, epochs=500) #xs: 입력데이타, ys= 정답 데이터
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)