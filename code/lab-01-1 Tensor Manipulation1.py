# import
import numpy as np
import torch

# 1D Array with Numpy
print('\n----1D Array with Numpy----')
t = np.array([0., 1., 2., 3., 4., 5., 6.]) # np array 생성
print(t)

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)
print('Number of element t: ', t.size)

print('t[0] t[1] t[-1] =', t[0], t[1], t[-1]) # Element
print('t[2:5] t[4:-1] =', t[2:5], t[4:-1] ) #Slincing (a이상 b미만)
print('t[:2] t[3:] =', t[:2], t[3:]) #Slincing


# 2D Array with Numpy
print('\n----2D Array with Numpy----')
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)
print('Number of element: ', t.size)

print(t[:, 1]) # 1열 원소 출력
print(t[:,1].size) #1열 원소 수
print(t[:, :-1]) #모든 행, 0 ~ -1미만 열


# Broadcasting => matrix 크기가 달라도 연산 해줌
print('\n----Broadcasting----')
# ex1)
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])
print(m1+m2)
# ex2)
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1+m2)


# Multiplication vs Matrix Multiplication
# (Broadcasting 기능은 자동적으로 수행되기 때문에 의도와 다르게 실행되는 상황을 주의해야 함)
print('\n----Multiplication vs Matrix Multiplication----')
# ex1) Multiplication
m1 = torch.FloatTensor([[1,2], [3,4]]) # 2X2
m2 = torch.FloatTensor([[1], [2]]) # 2X1
print(m1.matmul(m2)) # 2X1
# ex2) Matrix Multiplication (Broadcasting) : 원소별 곱으로 수행
m1 = torch.FloatTensor([[1,2], [3,4]]) # 2X2
m2 = torch.FloatTensor([[1], [2]]) # 2X1
print(m1 * m2) # 2X2
print(m1.mul(m2)) # 2X2


# Mean
# ex1)
t = torch.FloatTensor([1, 2])
print(t.mean())
# ex2, 3) dim=n: n차원 성분이 사라지도록 평균을 계산 !
t = torch.FloatTensor([[1,2], [3,4]])
print(t)
print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1)) # 참고


# Sum
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))


# Max and Argmax
t = torch.FloatTensor([[1,2],[3,4]])
print(t)
print(t.max(dim=0)) # value & index 함께 return
print(t.max(dim=0)[0]) # max value
print(t.max(dim=0)[1]) # max value index

print(t.max(dim=1))
print(t.max(dim=-1))