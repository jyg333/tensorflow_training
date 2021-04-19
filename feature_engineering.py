"""범주형 특성, 이산형 특성을 갖은 데이터들을 가지고 가장 적합한 데이터 표현을 찾는것-> 특성공학"""
#인구조사 데이터베이스을 예제로 사용
import pandas as pd
import os
import mglearn
from IPython.display import display

data = pd.read_csv(
    os.path.join(mglearn.datasets.DATA_PATH, "adult.data"),
header=None, index_col=False,
    names=['age','workclass','fnlwgt', 'education', 'education_num','marital_status','occupation',
'relationship','race','gender','capital_gain','capital_loss', 'hours_per_week', 'native_country','income']
)
data = data[['age', 'workclass','education','gender', 'hours_per_week','occupation','income']]
#예제를 위해 필요한 colum 만 선택
display(data.head(10))
# value_counts = pd.Series.value_counts()
print(data.gender.value_counts())
# Male and Female 값을 가지고 있어서 원핫인코딩을 하기 좋은 형태
print("원본 특성: \n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data) # get_dummies 함수를 사용해 데이터를 인코딩
print("get_dummies 후 특성: \n", list(data_dummies.columns))
display(data_dummies.head(10))

#data_dummies의 value 속성을 이용해 DataFrame을 Numpy배열로 바꿀 수 있다 -> 머신러닝 모델 학습 -> 타깃값 분리

#특성을 포함한 열 age에서 occupation_Transport-moving 까지 모든 열을 추출 -> 타갓을 뺀 모든 특성
features = data_dummies.loc[:,'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape: {}  y.shape: {}".format(X.shape, y.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train, y_train)
print("테스트 점수 : {:.2f}".format(logreg.score(X_test, y_test)))

#Data set을 linear model에 적용하려면, 범주형 변수에 원핫인코딩을 적용하고 연속형 변수의 스케일 조정
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    [("scaling", StandardScaler(),['age','hours_per_week']),
     ("onehot", OneHotEncoder(sparse=False), #sparse = False로 설정하면 희소행렬이 아니라 넘파이 배열로 반환
      ['workclass','education','gender','occupation'])])
#income을 제외한 모든열을 추출
data_features = data.drop('income', axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_features, data.income, random_state=0)
ct.fit(X_train)
X_train_trans = ct.transform(X_train)
print(X_train_trans.shape)
#pd.get_dummies를 사용했을 때와 마찬가지로 44개의 특성이 만들어졌다, 연속형 특성의 스케일을 조정했다는 부분이 다르다.

#logistic
logreg2 = LogisticRegression(max_iter=1000)
logreg2.fit(X_train_trans, y_train)

X_test_trans = ct.transform(X_test)
print(f'테스트 점수 : {logreg2.score(X_test_trans, y_test):.2f}')

