import config
import numpy as np
import scipy.sparse as sparse
import torch




def constructC(dxOpTemp, rhsDxOpTemp, dyOpTemp, rhsDyOpTemp, dlOpTemp, rhsDlOpTemp, psi, dxOpPsi, dyOpPsi, sqrtRa):
    dxPsi = torch.diag(torch.matmul(dxOpPsi , psi)[:, 0], 0)
    dyPsi = torch.diag(torch.matmul(dyOpPsi , psi)[:, 0], 0)
    C = sqrtRa * (torch.matmul(dyPsi , dxOpTemp) - torch.matmul(dxPsi , dyOpTemp )) - dlOpTemp
    rhsC  =  - rhsDlOpTemp + sqrtRa * (torch.matmul(dyPsi , rhsDxOpTemp)- torch.matmul(dyPsi , rhsDyOpTemp))
    return C,rhsC
