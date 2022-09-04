from re import S
import numpy as np
import torch

import config
import discretizationUtils as D1
import discretizationUtils1 as D2

nx = config.nx
ny = config.ny
data = torch.rand(ny,nx)
data_numpy = data.numpy()
a1 = D1.fieldToVec(data)
a2 = D2.fieldToVec(data_numpy)
print(a1.shape, a2.shape)
print(a1.numpy()-a2)

b1 = data.reshape(-1)
b2 = b1.numpy()

c1 = D1.vecToField(b1)
c2 = D2.vecToField(b2)
print(c1.shape,c2.shape)
print(c1.numpy()-c2)

