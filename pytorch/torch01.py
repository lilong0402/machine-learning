import torch
import numpy as np
# 创建一维张量
x = torch.tensor([1,2,3])

# 创建二维张量
y = torch.tensor([[1,2],[3,4]])

print(x)
print(y)
print("x的类型{}".format(type(x)))
print("x的类型{}".format(type(y)))

z = torch.from_numpy(np.array([1,2,3]))
a = torch.zeros(3, 4)  # 创建全零张量
b = torch.ones_like(x)  # 创建与给定张量形状相同且元素为1的张量
c = torch.randn(5, 6)  # 创建服从正态分布的随机张量

print("a:{},b:{},c:{},d:{}".format(a,b,c,z))

if torch.cuda.is_available():
    device = torch.device('cuda')
    d = torch.tensor([1,2,3],device=device)

    print(d.shape)