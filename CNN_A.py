import urllib.request
import zipfile
import numpy as np
from IPython.display import Image


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# 가위 바위 보 분
url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
urllib.request.urlretrieve(url, 'rps.zip')
local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

TRAINING_DIR = "tmp/rps/" #데이터 셋의 경로를 지정해 준다

training_datagen = ImageDataGenerator(
    rescale=1. / 255, #normalization
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest', # 빈 픽셀 값음 어떻게 체울것인가
    validation_split = 0.2 #80%의 데이터가 training에 할당되어 있다.
    )

#flow_from_directory로 이미지를 어떻게 공급해 줄 것인가를 지정
training_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                          batch_size = 128,
                                                          target_size = (150, 150),
                                                          class_mode = 'categorical',
                                                          subset='training',
                                                          #validation_split을 지정할 때만 사용
                                                         )
validation_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                            batch_size = 128,
                                                            target_size = (150, 150),
                                                            class_mode = 'categorical',
                                                            subset='validation',
                                                            )

model = Sequential([
                    Conv2D(64, (3, 3), activation='relu', input_shape = (150, 150, 3)),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3, 3), activation='relu', input_shape = (150, 150, 3)),
                    MaxPooling2D(2,2),
                    Conv2D(128, (3, 3), activation='relu', input_shape = (150, 150, 3)),
                    MaxPooling2D(2,2),
                    Conv2D(128, (3, 3), activation='relu', input_shape = (150, 150, 3)),
                    MaxPooling2D(2,2),
                    # Conv2D, MaxPooling2D 조합으로 층을 쌓습니다. 첫번째 입력층의 input_shape은 (150, 150, 3)으로 지정합니다.
                    Flatten(), #image data는 1치원으로 바꿔줭ㅑ한
                    Dropout(0.5), # prevent over_fitting
                    Dense(64, activation = "relu"),
                    Dense(3, activation = 'softmax'), #color image는 3개의 클래스를 갖는다.

                  ])
model.summary()

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics= 'acc')

checkpoint_path = "my_checkpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor="val_loss",
                             verbose=1)

model.fit(training_generator,
          validation_data = (validation_generator),
          steps_per_epoch = len(training_generator), # epoch이 넘어가지 않는 경우 len() 두가지 옵션을 넣어준다, 버전별로 상이하게 발생한다.
          validation_steps = len(validation_generator),
          epochs = 25,
          callbacks = [checkpoint]

          )

model.load_weights = (checkpoint_path) # 꼭 해주어야한다.