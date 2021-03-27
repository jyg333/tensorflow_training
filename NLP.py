import json
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Bidirectional,Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

with open('sarcasm.json') as f:
    datas = json.load(f)


sentences = []
labels = []

for data in datas:
    sentences.append(data['headline'])
    labels.append(data['is_sarcastic'])
for i in sentences[:5]:
    print([i])
# print(sentences[:5])
# print(labels[:5])

#train /validation 분리하기
train_size = 20000

train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

valid_sentences = sentences[train_size:]
valid_labels = labels[train_size:]

#tokenizer 의 정의
vocab_size = 1000
oov_tok = "<oov>"  #없는 단어를 치환하기 위한 단어, 절대 없을 법한 단어를 고른다
tokenizer = Tokenizer(num_words= vocab_size, oov_token='<oov>') #num_words: 모든 단어를 사용하지 않고 빈도수가 높은 1000개의 단어를 사용
tokenizer.fit_on_texts(train_sentences)  #토큰화

for key, value in tokenizer.word_index.items():
    print(f'{key} ----> {value}')
    if value == 25:
        break
"""단어사전에 들어있는 단어를 탐색해본다 """

# Replace sentences to sequences
train_sequences =tokenizer.texts_to_sequences(train_sentences)
valid_sequences =tokenizer.texts_to_sequences(valid_sentences)

word_index = tokenizer.word_index

# print(train_sequences[:5],'\n')

#pad_sequences 문장의 길이를 맞춰주는 전처리
# 1. 문장의 최대길이를 설정
# 2. 잘라낼 문장의 위치 설정
# 3. 채워넣을 문장의 위치 설정, pre, post 각각 앞과 뒤를 설정하는 명령어이다.
max_length = 120
trunc_type = 'post'
padding_type = 'post'

train_padded = pad_sequences(train_sequences,maxlen=max_length, truncating=trunc_type, padding=padding_type) # 변환된 sequences를 넣어준다, sentences를 넣지 않게 주의한다.
valid_padded = pad_sequences(valid_sequences,maxlen=max_length, truncating=trunc_type, padding=padding_type)
#print(train_padding.shape) #(20000, 120) 120단어를 갖은 20000 개의 문장 데이

#model은 list type은 받아들이 못함으로 label 값음 np.array로 변환해준다.

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)

embedding = 16 # one-hot encoding 했을 때 1000차원으로 표헌되는 단어를 16차원으로 변경

# Modeling
model = Sequential([
    Embedding(vocab_size, embedding, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
model.summary()

# 이후부터는 이전과 유사하다.
model.compile(optimizer='admam', loss='binary_crossentopy', metrics=['acc'])

checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

epochs =10

history = model.fit(train_padded, train_labels,
                    validation_data=(valid_padded, valid_labels),
                    callbacks=[checkpoint],
                    epochs=epochs
                    )
model.load_weights(checkpoint_path)
