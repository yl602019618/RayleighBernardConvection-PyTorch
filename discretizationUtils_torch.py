from re import S
import numpy as np
import torch

import config


def index2to1(y, x):
    '''
    input index of x and y
    first index y
    second index x
    range from 0, nx-1 , 0  ny-1
    output the index in flatten vector
    '''
    nx = config.nx
    ny = config.ny

    if (x < 0 or y < 0 or x >= nx or y >= ny):
        raise (IndexError)

    return (x * ny + y)


def index1to2y(k):
    '''
    input index of flatten vector
    
    range from 0 to nx*ny-1
    output the index in field y
    '''
    ny = config.ny
    nx = config.nx

    if (k > nx * ny - 1 or k < 0):
        raise (IndexError)

    return int(k % ny)


def index1to2x(k):
    '''
    input index of flatten vector
    
    range from 0 to nx*ny-1
    output the index in field x
    '''
    ny = config.ny
    nx = config.nx

    if (k > nx * ny - 1 or k < 0):
        raise (IndexError)

    return int(k / ny)


def fieldToVec(field):
    '''
    raval is flatten with non-copy
    '''
    vec = torch.ravel(field.t()).unsqueeze(-1)
    #vec.shape = (vec.size, 1)
    return vec

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def vecToField(vec):
    nx = config.nx
    ny = config.ny
    return reshape_fortran(vec,(ny,nx))
    #return (torch.reshape(vec, (ny, nx), order='F'))

# a = torch.arange(24).reshape(-1,6)
# print(torch.ravel(a.t()))

#  print(torch.reshape(a,[25],order = 'F'))
# a = np.arange(24).reshape(-1,6) 
# print(np.ravel(a,order='F'))
# a.shape = (a.size,1)
# print(a.shape)
# print(np.reshape(a, (2, 12), order='F'))
# a = torch.from_numpy(a)
# print(reshape_fortran(a,(2,12)))
# print(a.reshape(2,12).t().reshape(24).t().reshape(2,12))

