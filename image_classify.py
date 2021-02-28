import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


fashion_mnist = tf.keras.datasets.fashion_mnist # 10개의 클래스로 분류한다. 데이터 크기는 28 x 28만 사용한다.
(x_train, y_train), (x_valid, y_valid) =fashion_mnist.load_data()
print(x_train.min())
print(x_train.max())
print(y_train.shape)
print(y_valid.shape)

# Normalization: 모든 이미지 픽셀값ㅇㄹ 0~1 사이로 정의해준다 -> 수렴하는 속도가 빨라져서 성능개선
x_train = x_train/255
x_valid = x_valid/255
    # # Visualization : 샘플 데이터를 시각화 하는 코드이다. 시험에서는 다루지 않는다
    # fig, axes = plt.subplots(2, 5)
    # fig.set_size_inches = (10, 5)
    # for i in range(10):
    #     axes[i // 5, i % 5].imshow(x_train[i], cmap='gray')
    #     axes[i // 5, i % 5].set_title(str(y_train[i]), fontsize=15)
    #     plt.setp(axes[i // 5, i % 5].get_xticklabels(), visible=False)
    #     plt.setp(axes[i // 5, i % 5].get_yticklabels(), visible=False)
    #     axes[i // 5, i % 5].axis('off')
    #
    # plt.tight_layout()
    # plt.show()

model = Sequential([
    Flatten(input_shape=(28, 28)), #Dense layer는 2D 형태의 데이터를 받아들이 지못한다. Flatten을 이용해서 1차원 형태의 데이터로 바꿔준다
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.summary()
 # one_hot encdoing checking ,
print(y_train[0])
print(tf.one_hot(y_train[0], 10))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')
"""원핫인코딩이 안되어있을때의 loss , metrics로 학습시 정확도를 모니터 할 수 있다."""
# ModelCheckpoint 생성
# val_loss 를 기준으로 epoch 마 최적의 모델을 저장하기 위해, ModelCheckpoint를 만든다
# checkpoint_path는 모델이 저장될 파일 명을 설정합니다. ModelCheckpoint을 선언하고, 적절한 옵션 값을 지정합니다.
checkpoint_path = "my_checkpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor="val_loss",
                             verbose=1)

history = model.fit(x_train, y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=20,
                    callbacks=[checkpoint])

#학습이 완료된 후에는 반드시 load weight를 저장해 줘야한다.
model.load_weights(checkpoint_path)
#학습 후 검증
model.evaluate(x_valid, y_valid)
#학습 loss 시각화
plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, 21), history.history['loss'])
plt.plot(np.arange(1, 21), history.history['val_loss'])
plt.title('Loss / Val Loss', fontsize = 20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss','val_loss'], fontsize=15)
plt.show

plt.figure(figsize=(12, 9))
plt.plot(np.arange(1, 21), history.history['acc'])
plt.plot(np.arange(1, 21), history.history['val_acc'])
plt.title('Acc / Val Acc', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['acc', 'val_acc'], fontsize=15)
plt.show()
