#import numpy as np
import random
import os

#import autograd.numpy as np  # Thinly-wrapped numpy
#from autograd import grad
#from matplotlib import pyplot
from sklearn.decomposition import PCA
import math

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from model import Linear_Feature_Fusion
from data_generation import data_gen

def main():
    A,B,L = data_gen(10)
    alpha = A.shape[1]
    beta = B.shape[1]
    gamma = 2
    #P = torch.rand(alpha+beta,gamma)
    if not os.path.exists("data"):
        os.mkdir("data")
    outfile_a = open("data/A_values.txt",'w')
    for i in range(A.shape[0]):
        outfile_a.write(str(A[i].tolist()))
        outfile_a.write("\n")
    outfile_a.close()
    
    outfile_b = open("data/B_values.txt",'w')
    for i in range(B.shape[0]):
        outfile_b.write(str(B[i].tolist()))
        outfile_b.write("\n")
    outfile_b.close()
    
    outfile_l = open("data/L_values.txt",'w')
    for i in range(L.shape[0]):
        outfile_l.write(str(L[i].tolist()))
        outfile_l.write("\n")
    outfile_l.close()
    
    
    lamb = 0.2
    margin = 1#math.cos(math.pi/6)
    X = torch.cat((A,B),dim=1)
    M = []
    V = []
    for i in range(X.shape[0]):
        for j in range(i+1,X.shape[0]):
            if L[i] == L[j]:
                M.append((i,j))
                for k in range(X.shape[0]):
                    if L[k] != L[i]:
                        V.append((i,j,k))
    
    model = Linear_Feature_Fusion(X,M,V,gamma,margin,lamb)
    
    iterations = 100
    break_point = 0.005
    rate = 10e-2

    best_loss = model.loss()
    best_P = model.P
    print("Initial loss:",best_loss)
    P_history = []
    losses = []
    optim = torch.optim.SGD(model.parameters(), lr=rate)#, momentum=0.9)
    for i in range(iterations):
        loss = model.loss()
        if loss < best_loss:
            best_loss = loss
            best_P = model.P
        losses.append(loss)
        P_history.append(str(model.P.tolist()))
        loss.backward()
        optim.step()
        if i%10 == 0:
            print("Iteration",i)
    print("final loss:",model.loss())
    
    outfile_p = open("data/P_values.txt",'w')
    for P_value in P_history:
        outfile_p.write(P_value)
        outfile_p.write("\n")
    outfile_p.close()
    
    outfile_p_t = open("data/best_P_value_transpose.txt",'w')
    P_final_t = best_P.T
    P_final_t = str(P_final_t.tolist())
    outfile_p_t.write(P_final_t)
    outfile_p_t.close()
    
    outfile_loss = open("data/loss_values.txt",'w')
    for loss_value in losses:
        outfile_loss.write(str(loss_value.tolist()))
        outfile_loss.write("\n")
    outfile_loss.close()

if __name__ == "__main__":
    main()
