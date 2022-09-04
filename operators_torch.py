import config
import numpy as np
import scipy.sparse as sparse
import torch
def dyOp():
    """Create discrete operator on 2d grid for d( . ) / dy using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries. Doesn't use any boundary conditions (and is therefore unused in the
    project).

    Status: finished on 21 March.

    :return: scipy.CSR_matrix containing linear operator
    """
    nx = config.nx
    ny = config.ny
    dy = config.dy

    plus1 = torch.ones((ny,)) # [0,1,1,...,1,2]
    plus1[-1] = 0
    plus1[0] = 2
    mid = torch.zeros((ny,)) # [-2,0,0,...,0,2]
    mid[0] = -2
    mid[-1] = 2
    min1 = -torch.ones((ny,)) # [-1,-1,...,-1,-2,0]
    min1[-1] = 0
    min1[-2] = -2

    plus1 = torch.tile(plus1, (nx,))
    mid = torch.tile(mid, (nx,))
    min1 = torch.tile(min1, (nx,))

    result =  (torch.diag(plus1[:-1],1) + torch.diag(mid,0) + torch.diag(min1[:-1],-1))/(2.0*dy)
    return result


# Temperature operators
def dyOpTemp(bcDirArray):
    """Create discrete operator on 2d grid for dT / dy using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate nodes from the
    LHS. To calculate the dT / dy field one would perform A @ T - rhs.

    Status: finished on 21 March.

    :return: tuple of (scipy.CSR_matrix containing linear operator, numpy.ndarray of rhs vector)
    """
    nx = config.nx
    ny = config.ny
    dy = config.dy

    # First construct the right hand side from one-sided difference formulas for y = 0 & y = ymax and the Dirichlet BC.
    rhs = torch.zeros((nx * ny,))

    for ix in range(nx):
        rhs[0 + ny * ix] = bcDirArray[0][ix] / dy  # y = 0
        rhs[ny * (ix + 1) - 1] = -  bcDirArray[1][ix] / dy  # y = ymax
    rhs = rhs.unsqueeze(1)  # Make it an explicit column vector

    # We use the central difference formula for all the inner derivatives ...
    plus1 = torch.ones((ny,))
    mid = torch.zeros((ny,))
    min1 = -torch.ones((ny,))

    # ... and again the one-sided difference formulas for y = 0 & y = ymax
    plus1[-1] = 0
    plus1[0] = 2
    min1[-1] = 0
    min1[-2] = -2

    # Repeat along x = 0, 1, 2, ...
    plus1 = torch.tile(plus1, (nx,))
    mid = torch.tile(mid, (nx,))
    min1 = torch.tile(min1, (nx,))

    # Construct a sparse matrix from diagonals
    A =  (torch.diag(plus1[:-1],1) + torch.diag(mid,0) + torch.diag(min1[:-1],-1))/(2.0*dy)
    return A, rhs


def dxOpTemp(bcNeuArray):
    """Create discrete operator on 2d grid for dT / dx using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate nodes from the
    LHS. To calculate the dT / dx field one would perform A @ T - rhs.

    Status: finished on 20 March.

    :return: tuple of (scipy.CSR_matrix containing linear operator, numpy.ndarray of rhs vector)
    """
    nx = config.nx
    ny = config.ny
    dx = config.dx

    rhs = torch.zeros((nx * ny,))
    rhs[0:ny] = - bcNeuArray[0]
    rhs[-ny:] = - bcNeuArray[1]
    rhs = rhs.unsqueeze(-1)  # Make it an explicit column vector

    plus1 = torch.ones((ny,))
    mid = torch.zeros((ny,))
    min1 = -torch.ones((ny,))

    plus1 = torch.tile(plus1, (nx - 1,))
    plus1[0:ny] = 0

    mid = torch.tile(mid, (nx,))

    min1 = torch.tile(min1, (nx - 1,))
    min1[-ny:] = 0

    #A = sparse.csr_matrix(sparse.diags([plus1[:], mid, min1[:]], [ny, 0, -ny]) / (2.0 * dx))
    A =  (torch.diag(plus1[:],ny) + torch.diag(mid,0) + torch.diag(min1[:],-ny))/(2.0*dx)

    return A, rhs

def dlOpTemp(bcDirArray, bcNeuArray):
    """Create discrete operator on 2d grid for d²T / dx² + d²T / dy² using central and one-side difference formulas.
    The one-sided formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate
    nodes from the LHS. Neumann conditions are used to simplify calculation of d²T/dx². To calculate the dT / dx field
    one would perform A @ T - rhs.

    Status: finished on 21 March.

    :return: tuple of (scipy.CSR_matrix containing linear operator, numpy.ndarray of rhs vector)
    """
    nx = config.nx
    ny = config.ny
    dx = config.dx
    dy = config.dy

    rhs = torch.zeros((nx * ny,))
    rhs = rhs.unsqueeze(-1)

    # This is the center of the stencil in centered difference formula
    mid = -2 * torch.ones((ny,)) / (dx ** 2) - 2 * torch.ones((ny,)) / (dy ** 2)
    # But the points at y = 0 and y = ymax are using a one-sided difference
    mid[0] = -2 / (dx ** 2) + 1 / (dy ** 2)
    mid[-1] = mid[0]
    mid = torch.tile(mid, (nx,))
    # And for x=0 and x=xmax we also have the modified formula for the x coordinate (note that the formula actually is
    # very similar as the centered difference, although accuracy is lower.)
    rhs[0:ny] = bcNeuArray[0] / (0.5 * dx)
    rhs[-ny:] = -bcNeuArray[1] / (0.5 * dx)

    # Now we generate all the diagonals. Plus/min <number> x/y stands for placement in the 2D grid. Dirichlet conditions
    # are handled later.

    # Two sided difference formula for d/dy to next point
    plus1y = torch.ones((ny,)) / (dy ** 2)
    plus1y[0] = -2 / (dy ** 2)  # one sided
    plus1y[-1] = 0  # not part of the scheme
    plus1y = torch.tile(plus1y, (nx,))
    plus1y = plus1y[:-1]

    # Two sided difference formula for d/dy to previous point
    min1y = torch.ones((ny,)) / (dy ** 2)
    min1y[-2] = -2 / (dy ** 2)  # one side
    min1y[-1] = 0  # not part of the scheme
    min1y = torch.tile(min1y, (nx,))
    min1y = min1y[:-1]

    plus2y = torch.zeros((ny,))
    plus2y[0] = 1 / (dy ** 2)
    plus2y = torch.tile(plus2y, (nx,))
    plus2y = plus2y[:-2]

    min2y = torch.zeros((ny,))
    min2y[-3] = 1 / (dy ** 2)
    min2y = torch.tile(min2y, (nx,))
    min2y = min2y[:-2]

    plus1x = torch.ones((ny,)) / (dx ** 2)
    plus1x = torch.tile(plus1x, (nx - 1,))
    plus1x[:ny] = 2 / (dx ** 2)

    min1x = torch.ones((ny,)) / (dx ** 2)
    min1x = torch.tile(min1x, (nx - 1,))
    min1x[-ny:] = 2 / (dx ** 2)


    A =  (torch.diag(min1x,-ny) + torch.diag(min2y,-2) + torch.diag(min1y,-1)\
        +torch.diag(mid,0)+torch.diag(plus1y,1)+torch.diag(plus2y,2)+torch.diag(plus1x,ny))

    #A = sparse.csr_matrix(sparse.diags([min1x, min2y, min1y, mid, plus1y, plus2y, plus1x], [-ny, -2, -1, 0, 1, 2, ny]))

    toKeep = torch.ones((nx * ny,))

    # Now we need to construct the RHS for the Dirichlet points
    for ix in range(nx):
        rhsPart = (A[:, ix * ny] * bcDirArray[0][ix]).reshape(nx * ny, 1)
        rhs = rhs - rhsPart.clone()
        toKeep[ix * ny] = 0
        rhsPart = (A[:, (ix + 1) * ny - 1] * bcDirArray[1][ix]).reshape(nx*ny , 1)
        rhs = rhs - rhsPart.clone()
        toKeep[(ix + 1) * ny - 1] = 0

    A = torch.matmul(A , torch.diag(toKeep, 0)) # Deleting Dirichlet columns

    return A, rhs


# Streamfunction operators
def dxOpStream():
    """Create discrete operator on 2d grid for dPsi / dx using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate nodes from the
    LHS. To calculate the dPsi / dx field one would perform A @ T. I omitted including the Dirichlet conditions because
    this matrix is never used to solve for Psi.

    Status: finished on 21 March.

    :return: scipy.CSR_matrix containing linear operator
    """
    nx = config.nx
    ny = config.ny
    dx = config.dx

    plus1 = torch.ones((ny,))
    mid = torch.zeros((ny,))
    min1 = -torch.ones((ny,))

    plus1 = torch.tile(plus1, (nx - 1,))
    plus1[0:ny] = 2

    mid = torch.tile(mid, (nx,))
    mid[0:ny] = -2
    mid[-ny:] = 2

    min1 = torch.tile(min1, (nx - 1,))
    min1[-ny:] = -2

    #A = sparse.csr_matrix(sparse.diags([plus1[:], mid, min1[:]], [ny, 0, -ny]) / (2.0 * dx))
    A = (torch.diag(plus1[:],ny) + torch.diag(mid,0) + torch.diag(min1[:],-ny)) /(2.0*dx)
    return A


def dyOpStream():
    """Create discrete operator on 2d grid for dPsi / dy using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate nodes from the
    LHS. To calculate the dPsi / dy field one would perform A @ T. I omitted including the Dirichlet conditions because
    this matrix is never used to solve for Psi. It is of course then identical to dyOp().

    Status: finished on 21 March.

    :return: scipy.CSR_matrix containing linear operator
    """
    nx = config.nx
    ny = config.ny
    dy = config.dy

    plus1 = torch.ones((ny,))
    plus1[-1] = 0
    plus1[0] = 2
    mid = torch.zeros((ny,))
    mid[0] = -2
    mid[-1] = 2
    min1 = -torch.ones((ny,))
    min1[-1] = 0
    min1[-2] = -2

    plus1 = torch.tile(plus1, (nx,))
    mid = torch.tile(mid, (nx,))
    min1 = torch.tile(min1, (nx,))

    #A = sparse.csr_matrix(sparse.diags([plus1[:-1], mid, min1[:-1]], [1, 0, -1]) / (2.0 * dy))
    A = (torch.diag(plus1[:-1],1) + torch.diag(mid,0) + torch.diag(min1[:-1],-1)) /(2.0*dy)
    return A


def dlOpStreamMod():
    """Create discrete operator on 2d grid for d²Psi / dy² using central and Dirichlet boundary conditions. It is not
    the traditional operator, as some entri=es are modified to simplify the solving of equation 1 from the project.

        Status: finished on 21 March.

        :return: scipy.CSR_matrix containing linear operator
        """
    nx = config.nx
    ny = config.ny
    dx = config.dx
    dy = config.dy

    mid = -2 * torch.ones((ny,)) / (dx ** 2) + -2 * torch.ones((ny,)) / (dy ** 2)
    mid[0] = 1
    mid[-1] = 1

    plus1x = torch.ones((ny,)) / (dx ** 2)
    plus1x[0] = 0
    plus1x[-1] = 0
    min1x = torch.ones((ny,)) / (dx ** 2)
    min1x[0] = 0
    min1x[-1] = 0

    plus1y = torch.ones((ny,)) / (dy ** 2)
    plus1y[0] = 0
    plus1y[-1] = 0
    min1y = torch.ones((ny,)) / (dy ** 2)
    min1y[0] = 0
    min1y[-1] = 0

    # Assemble large scale x
    plus1x = torch.tile(plus1x, (nx - 1,))
    plus1x[0:ny] = 0
    min1x = torch.tile(min1x, (nx - 1,))
    # min1x[0:ny] = 0
    min1x[-ny:-1] = 0

    # Assemble large scale y
    plus1y = torch.tile(plus1y, (nx,))
    plus1y = plus1y[0:-1]
    plus1y[0:ny] = 0
    plus1y[-ny:] = 0
    min1y = torch.tile(min1y, (nx,))
    min1y = min1y[1:]
    min1y[0:ny] = 0
    min1y[-ny:-1] = 0

    mid = torch.tile(mid, (nx,))
    mid[0:ny] = 1
    mid[-ny:-1] = 1

    #return sparse.csc_matrix(sparse.diags([mid, plus1y, min1y, plus1x, min1x], [0, 1, -1, ny, -ny]))
    return torch.diag(mid, 0) + torch.diag(plus1y , 1) + torch.diag(min1y,-1)+ torch.diag(plus1x,ny) + torch.diag(min1x , -ny)



# A1 = dlOpStreamMod().numpy()
# A2 = dlOpStreamMod1() 
# print(np.max(np.abs(A1-A2)))

# dy1 = dyOp1().A
# dy2 = dyOp().numpy()
# print(np.max(dy1-dy2))
# print(dy1.shape,dy2.shape)
#a_t = torch.arange(config.nx*config.ny).reshape(config.nx,-1)
# a_t = torch.rand(config.nx,config.ny,dtype = torch.float32)
# a = a_t.numpy()
# b_t = torch.rand(config.ny,dtype = torch.float32)
# b = b_t.numpy()
# A1 ,rhs1 = dlOpTemp(a_t,b_t)
# A2 ,rhs2 = dlOpTemp1(a,b)
# # A2,rhs2 = dxOpTemp1(a)
# print('A1',A1,'A2',A2,np.max(A1.numpy()-A2.A),'rhs1',rhs1.numpy(),rhs2)
# # A1, rhs1 = dyOpTemp(a_t)
# A , rhs = dyOpTemp1(a)
# print(rhs1.numpy()-rhs)
# print(rhs1,rhs)

