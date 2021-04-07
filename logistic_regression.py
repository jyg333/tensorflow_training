import pandas as pd
import numpy as np

fish = pd.read_csv('http://bit.ly/fish_csv')
#print(fish.head(10)) #처음 10행 출력
#print(pd.unique(fish['Species'])) # unique() 함수를 통해서 Species열에 어떤 고유한 값이 있는지 추출
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
'''데이터 프레임에서 Species열을 타깃으로 만들고 나머지 5가지 열은 입력 데이터로 사용'''

fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# sklearn의 클래스를 사용해서 훈련 세트와 테스트 세트를 표준화하는 전처리
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃 분류기를 사용하여 확률예측

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3) #최근접 이웃 개수 =3
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled,train_target)) #0.89
print(kn.score(test_scaled,test_target)) #0.85

# target_data를 만들때 fish['Species']를 사용했기 떄문에 훈련/테스트 세트에 7개의 클래스가 있다.-> 다중분류

print(kn.classes_)
print(kn.predict(test_scaled[:5]))
# 5번째 까지의 생선의 종류를 예측한 결과를 확률의 분표로 비교해 본다
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals =4)) #소수 4번째 자리까지 표기

distances, indexs = kn.kneighbors(test_scaled[3:4]) #kneighbor() 매서드의 입력은 2차원 배열어야한다. 넘파이의 슬라이싱 연사자(:) 사용
print(train_target[indexs])
# print(distances)
# print(indexs)

#logistic regression
#선형회귀와 동일하게 선형 방정식으로 학습하는 알고리즘, 특성은 늘어났지만 다중회귀를 위한 성형방정식
#sigmoid 구현
import matplotlib.pyplot as plt

# z = np.arange(-5, 5, 0.1)
# phi = 1/(1+np.exp(-z))
# plt.plot(z, phi)
# plt.xlabel('z')
# plt.ylabel('phi')
# plt.show()

#boolean indexing
# char_arr = np.array(['A','B','C','D','E'])
# print(char_arr[[True, False, True, False, False]])

#bream amd smelt are True the others are False
bream_smelt_indexes = (train_target =='Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

#Logistic Regression Model Training
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))
"""샘플마다 2개의 열이 출력되는데, 첫번째 열은 0에대 확률이고 두번째 열은 1에대 확률이다.
sklearn은 타깃값을 알파벳 순으로 정렬하기 때문에 bream == 0, smelt == 1 이다."""

decisions = lr.decision_function(train_bream_smelt[:5]) #decsion_function()으로 가중치를 곱한 출력값을 볼수 있다.
print(decisions)
#출력값을 sigmoid함수를 통과시켜 확률을 얻을 수 있다.
from scipy.special import expit
print(expit(decisions))

lr2 = LogisticRegression(C=22, max_iter= 1000)
lr2.fit(train_scaled,train_target)
print(lr2.score(train_scaled, train_target))
print(lr2.score(test_scaled, test_target))

#5개의 샘플에 대한 출력
print(lr2.predict(test_scaled[:5]))
proba = lr2.predict_proba((test_scaled[:5]))
print(np.round(proba, decimals = 3))

print(lr2.classes_) # checking infroamtion of classes
"""데이터는 5개의 특성을 사용하여 7개의 클래스를 계산한다
높은 z 값을 출력하는 클래스가 예측 클래스가 된다. 특성에 가중치를 곱한 값인 z값은 어떤 숫자든 될 수 있다.
그 값을 확률적으로 변환하기위한 함수가 softmax이다. 이진 분류일때는 sigmoid 함수를 사용하지만, 3개 이상일때
부터는 softmax함수를 사용한다"""

decision = lr2.decision_function(test_scaled[:5])
"""AxisError: axis 1 is out of bounds for array of dimension 1
lr -> lr2 로 바꿔주고 에러 해결 lr2의 C 매개변수로 규제를 완화함"""
from scipy.special import softmax
proba = softmax(decision, axis=1) #axis =1, 소프트맥스를 계산 할 축을 지정, 각 행에대한 소프트맥스 계산
print(np.round(proba, decimals =3)) #최종 확률 출력
