# tensorflow regression predict

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
""" Sequential 모델은 각각의 레이어에 하나의 입력과 하나의 출력이 있는 일반 레이어 스택에 적"""

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)

model = Sequential([Dense(1, input_shape= [1])])
model.compile(optimizer='sgd', loss = 'mse') # 확률경사 하강법을 사용, 평균제곱 오차사용
model.fit(xs, ys, epochs= 1200, verbose = 1) # verbose 기본값은 1인데 로그기록을 출력하지 않고 싶을 떄사용
#model.summary()


print(model.predict([10.0]))