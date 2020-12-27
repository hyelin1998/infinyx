# import
import numpy as np
import torch

# View(Reshape) => 원하는 형태로 Tensor의 크기를 변환
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8],[9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape) # 2X2X3

print(ft.view([-1, 3])) # 2차원인데, 첫번째 차원의 크기는 모르고 두번째 차원의 크기는 3
print(ft.view([-1, 3]).shape) # 4X3 ( =12, element 수 같음)

print(ft.view([-1, 1, 3])) # 3차원인데, 첫번째 차원은 auto, 두번째는 크기 1, 세번째는 3
print(ft.view([-1, 1, 3]).shape) # 4X1X3 ( =12, element 수 같음)


# Squeeze
ft = torch.FloatTensor([[0],[1],[2]])
print(ft)
print(ft.shape) # 3X1

print(ft.squeeze())
print(ft.squeeze().shape) # 3 ( dimsize=1 인 차원을 없애줌)


# Unsqueeze (Squeeze 반대)
ft = torch.Tensor([0, 1, 2])
print(ft.shape) # 3
print(ft.unsqueeze(0)) # => ft.view(1, -1)과 같은 작동
print(ft.unsqueeze(0).shape) # 1X3 (0번째 차원 추가)
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape) # 3X1 (1번째 차원 추가)


# Type Casting
lt = torch.LongTensor([1, 2, 3, 4]) # long type
bt = torch.ByteTensor([True, False, False, True]) # boolian type
print(lt)
print(lt.float()) # float type 으로 형변환
print(bt)
print(bt.long()) # long type 으로 형변환
print(bt.float()) # float type 으로 형변환


# Concatenate
x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])
print(torch.cat([x,y], dim=0)) # 0번째 dim 이 늘어나도록 concat
print(torch.cat([x,y], dim=1)) # 1번째 dim 이 늘어나도록 concat


# Stacking
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z])) # 0번째 dim 이 늘어나도록 쌓음
print(torch.stack([x, y, z], dim=1)) # 1번째 dim 이 늘어나도록 쌓음
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0 ))


# Ones and Zeros
x = torch.FloatTensor([[0,1,2],[2,1,0]])
print(x)
print(torch.ones_like(x)) # 같은 크기에, 1로 초기화
print(torch.zeros_like(x)) # 같은 크기에, 0으로 초기화


# In-place Operation => 메모리를 새로 선언하지 않고, 기존의 메모리 사용
x = torch.FloatTensor([[1,2],[3,4]])
print(x.mul(2.)) # 원본값 바꾸지 않음 (x * 2)
print(x)
print(x.mul_(2.)) # 원본값을 수정 (in place operation)
print(x)