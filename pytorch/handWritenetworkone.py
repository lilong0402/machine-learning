'''以下是全部所需库，pickle用于载入数据集,数据集很简单，后面会提到'''
import numpy as np
import math,pickle,time
import matplotlib.pyplot as plt
from collections import defaultdict
from abc import ABC,abstractmethod,abstractproperty

'''对于拷贝传递返回值的成员方法,导数也拷贝生成'''


def add_grad(func):
    def inner(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        ret.detach = False
        ret.grad = np.zeros(ret.shape)
        return ret

    return inner


'''对于引用传递返回值的成员方法,导数也取引用'''
'''
修改了add_grad_inplace,
之前的方法其实并没有让导数的内容和原数组一一对应,本篇未用到这个方法,
但是后续会更新的"手写CNN"里会用到
'''


def add_grad_inplace(func):
    def inner(self, *args, **kwargs):
        grad = self.grad
        ret = Tensor(func(self, *args, **kwargs))
        ret.grad = getattr(grad, func.__name__)(*args, **kwargs)
        return ret

    return inner


class Tensor(np.ndarray):
    '''传入元组会生成元组形状的随机矩阵,其余参数仅进行拷贝封装'''
    '''
	修改了__new__方法的if,else条件判断,代码更简洁和通用,注释掉的是修改前的
	'''

    def __new__(cls, input_array, requires_grad=True):
        if type(input_array) == tuple:
            obj = np.random.randn(*input_array).view(cls)
        else:
            obj = np.asarray(input_array).view(cls)
        obj.grad = np.zeros(obj.shape)
        return obj

    #    def __new__(cls,input_array,requires_grad=True):
    #        obj=np.asarray(input_array).view(cls) \
    #        if type(input_array) in (list,Tensor,np.ndarray) \
    #        else np.random.randn(*input_array).view(cls)
    #        obj.grad=np.zeros(obj.shape)
    #        return obj
    '''我一通乱装饰，实际上真正用到的没这么多，选择自己需要的进行装饰就好'''

    @add_grad
    def mean(self, *args, **kwargs):
        return super().mean(*args, **kwargs)

    @add_grad
    def std(self, *args, **kwargs):
        return super().std(*args, **kwargs)

    @add_grad
    def sum(self, *args, **kwargs):
        return super().sum(*args, **kwargs)

    @add_grad
    def __add__(self, *args, **kwargs):
        return super().__add__(*args, **kwargs)

    @add_grad
    def __radd__(self, *args, **kwargs):
        return super().__radd__(*args, **kwargs)

    @add_grad
    def __sub__(self, *args, **kwargs):
        return super().__sub__(*args, **kwargs)

    @add_grad
    def __rsub__(self, *args, **kwargs):
        return super().__rsub__(*args, **kwargs)

    @add_grad
    def __mul__(self, *args, **kwargs):
        return super().__mul__(*args, **kwargs)

    @add_grad
    def __rmul__(self, *args, **kwargs):
        return super().__rmul__(*args, **kwargs)

    @add_grad
    def __pow__(self, *args, **kwargs):
        return super().__pow__(*args, **kwargs)

    @add_grad
    def __rtruediv__(self, *args, **kwargs):
        return super().__rtruediv__(*args, **kwargs)

    @add_grad
    def __truediv__(self, *args, **kwargs):
        return super().__truediv__(*args, **kwargs)

    @add_grad
    def __matmul__(self, *args, **kwargs):
        return super().__matmul__(*args, **kwargs)

    @add_grad
    def __rmatmul__(self, *args, **kwargs):
        return super().__rmatmul__(*args, **kwargs)

    @add_grad_inplace
    def reshape(self, *args, **kwargs):
        return super().reshape(*args, **kwargs)

    @add_grad_inplace
    def __getitem__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)

    @property
    def zero_grad_(self):
        self.grad = np.zeros(self.grad.shape)


'''定义网络所需的函数'''
'''
现在这种写法更加简便,其实受益于继承关系,直接Tensor(np.exp(x))也可以,
但是有风险会报错
'''


def exp(x):
    return Tensor(np.exp(np.array(x)))


def log(x):
    return Tensor(np.log(np.array(x)))
