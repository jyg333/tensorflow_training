from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import export_graphviz
from IPython.display import display
"""유방암 데이터셋 활용"""

cancer = load_breast_cancer()
print(cancer)
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("훈련세트의 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트세트의 정확도: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=42) #트리의 깊이가 무한정 깊어져 과적합되지 않도록 제한
tree.fit(X_train, y_train)
print("훈련세트의 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트세트의 정확도: {:.3f}".format(tree.score(X_test, y_test)))
#훈련세트의 정확도는 떨어지지만, 테스트 세트의 정확도는 증가.

export_graphviz(tree, out_file="tree.dot", class_names=["악성", "양성,"],
                feature_names=cancer.feature_names, impurity=False, filled=True) #filled =Ture 는 노트의 클래스가 구분되도록 색
import graphviz
with open("tree.dot") as f:
    dot_graph =f.read()
    # display(graphviz.Source(dot_graph))
    dot = graphviz.Source(dot_graph)
    dot.format = 'png'
    dot.render(filename = 'tree')
    display(dot)

import matplotlib.pyplot as plt
import numpy as np
import mglearn


#Feature Importance
def plot_feature_importance_cancer(model):
    n_features = cancer.data.shape[1]
    print(tree.feature_importances_)
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성의 중요도")
    plt.ylabel("특성")
    plt.ylim(-1,n_features)
    plt.show()

# plot_feature_importance_cancer(tree)

"""DecisionTreeRegression과 LinearRegression을 비교해보기로 한다. 2000년 전의 가격으로 2000년 후의 가격을 예측해 보기로 한다.
가격값을 로그로 바꿨기 때문에 비교적 선형적인 관계를 갖는다
Ram_prices datasets을 사용했다."""
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import multiclass

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

#날짜 특성을 이용
X_train = data_train.date[:, np.newaxis]
y_train = np.log(data_train.price)
print(X_train.dtype)
print(y_train.dtype)
#Error message : (ValueError: Unknown label type: 'continuous')
y_train = y_train.astype('int')
print(y_train.dtype)
# y_train = multiclass.type_of_target(y_train.astype('int'))

tree = DecisionTreeClassifier().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

#전체 기간에 대한 예측
X_all = ram_prices.date[:, np.newaxis]

predict_tree = tree.predict(X_all)
predict_linear = linear_reg.predict(X_all)

# 예측한 값을 로그 스케일로 반환
price_tree = np.exp(predict_tree)
price_linear = np.exp(predict_linear)

plt.semilogy(data_train, data_train.price, label ="훈련 데이터")
plt.semilogy(data_test, data_test.price, label ="테스트 데이터")
plt.semilogy(ram_prices.date, price_tree, label ="트리 예측")
plt.semilogy(ram_prices.date, price_linear,label = "선형회귀 예측")
plt.xlim(1950, 2020)
plt.legend()
plt.show()

'''트리 예측과 선형모델의 예측은 차이를 보인다.
2000년 이후 트리 모델은 복잡도에 제한을 두지 않아서 전체 데이터셋을 모두 기억하기 때문인데, 트리 모델은 훈련 데이터 밖의 새로운 
데이터를 예측하는 능력은 없다.'''





