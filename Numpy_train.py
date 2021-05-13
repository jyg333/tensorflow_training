import numpy as np

list_data = [1, 2, 3]
print(list_data)

array = np.array(list_data)
print(array)
print(array.size)
print(array.dtype)
print(array[2])

# from 0 to 3
array1 = np.arange(4)
print(array1)
#Make (4x4)  matrix float type data
array2 = np.zeros((4, 4), dtype=float)
print(array2)

array3 = np.ones((3, 3 ), dtype=str)
print(array3)

# from 0 to 9,random int
array4 = np.random.randint(0, 10, (3, 3))
print(array4)

# 평균이 0이고, 표준편차가 1인 표준 정규를 띄는 배열
array5 =np.random.normal(0, 1, (3, 3))
print(array5)

#배열을 합치기, concatenate(연쇄시키다) function 사용
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
array3 = np.concatenate([array1, array2])
print(array3.shape)
print(array3)

"""numpy는 형태를 자유롭게 바꿀 수 있다는 장점이 있다. 이것이 ML에서 python이 강점을 나타내는 이유중 하나이다."""
array4 = np.array([1, 2, 3, 4])
array5 = array4.reshape((2, 2))
print(array5)

array6 = np.arange(4).reshape(1, 4)
array7 = np.arange(8).reshape(2, 4)
print(array6)
print(array7)
# 세로축 기준으로 합치기 위해, axis = 0
array8 = np.concatenate([array6, array7])
print(array8)

#배열 나누기, split

left, right = np.split(array7 , [2], axis =1) #2번째 열을 기준으로 나눠준다
print(left.shape)
print(right.shape)

"""numpy의 연산, 기본적으로 4칙연산을 제공한다."""
array9 = np.random.randint(1, 10, size = 4).reshape(2, 2)
print(array9)
array9 = array9 * 10
print(array9)

#서로 다른 형태의 배열의 연산이 가능하다.
#Broadcasting, 형태가 다른 배열 계산이 가능하도록 동적으로 변환
array10 = np.arange(4).reshape(2, 2)
array11 = np.arange(2) # (1, 2)
array12 = array10 + array11
print(array12)

array13 = np.random.randint(1, 8, (2, 4))
array14 = np.arange(0, 8).reshape(2, 4)
array15 = np.concatenate([array13,  array14], axis = 0)
array16 = np.arange(0,4).reshape(4, 1)


print(array13)
print(array14,"\n")
print(array15)
print(array15 + array16)
"""masking calculation, 
마스킁: 각 원소에 대해서 어떠한 조건을 만족하는지 체크한다. """
array17 = np.arange(16).reshape(4, 4)
array_masking = array17 > 10
print(array_masking)
#마스킹을 통해서, 특정 조건을 만족하는 원소에 대해서 어떠한 작업을 수행하도록 명령하는 것이 가능하다.
#exam, 픽셀의 밝기 특정 점을 넘는 원소에 대해서 그 값을 바꾸겠다.
array17[array_masking] = 100
print(array17)

"""numpy의 집계함수"""
array18 = np.random.randint(1, 100, (4, 4))
print(array18)
print("Maximum : ",np.max(array18))
print("Minimum : ",np.min(array18))
print("Sum : ",np.sum(array18))
print("Average : ",np.mean(array18))
#특정한 열이나 행에 대한 값을 출력 할 수 있다.
print("합계 : ", np.sum(array18, axis=0))

"""Numpy data, save and load """
array19 = np.arange(10)
np.save("saved.npy\n", array19)

load_np = np.load("saved.npy")
print(load_np)
#복수 객체도 가능하다.
array20 = np.arange(10, 20)
np.savez("saved.npz", array19 = array19,array20 = array20)

load_npz = np.load("saved.npz")
result1 = load_npz['array19']
result2 =load_npz['array20']
print(result1)
print(result2)

"""Numpy의 정렬"""
#오름차순
array21 = np.random.randint(100, size = 20)
array21.sort()
print(array21)
print(array21[::-1]) #reversed
#각 열을 기준으로 정렬
array22 = np.random.randint(100, size = 20)
array22 = array21.reshape(4, 5)
print(array22)
array21.sort(axis = 0)
print(array22)

#균일한 간격으로 데이터 생성
array23 = np.linspace(0, 10, 5)
print(array23)

# 배열 개체 복사
array24 = np.arange(10)
array25 = array24
array25[0]= 100
print(array24)
array26 = np.arange(10)
array27 = array26.copy() # array26의 객체는 바뀌지 않는다.
array25[0] = 100
print(array26)

#중복된 원소를 제거
array28 = np.array([ 1,1,2,2,4,4,5,5,7,8,9,9])
print(np.unique(array28 ))
