"""
This file contains implementations of the Optimal Sparce Input Features problem described in
Section 7.1.2 of the manuscript. The file can be run directly, with the following parameters.
Parameters can be changed on lines 16-18.
    
    Parameters:
        modelName (str): name of the file containing model parameters
        targetClass (int): target class to maximize, can be 0-9
        nPartitions (int): number of partitions to use
        epsilon (real): maximum l1-norm defining perturbations
"""

import numpy as np
import gurobipy as gb
from pathlib import Path

modelName = 'mnist_dense3x50_1'
targetClass = 0
nPartitions = 2
epsilon = 5

NNparams = np.load('models/'+str(modelName))
nLayers = 3 

# Define rule for partitioning weights
def getSplit(m, nSplits):
    return np.array_split(np.argsort(m), nSplits)

#BEGIN OPTIMIZATION MODEL
model = gb.Model()
x = {}; y = {}; z2 = {}; sig = {};

# Create input nodes
x[0] = {}; z2[0] = {}; sig[0] = {}
for i in range(NNparams['m1'].shape[1]):
    x[0][i] = model.addVar(0, 1, name='x_' + str(0) + '_' + str(i))

model.update()
model.addConstr(sum(x[0][i] for i in range(NNparams['m1'].shape[1])) <= epsilon)

for ind in range(nLayers):
    x[ind + 1] = {}; z2[ind + 1] = {}; sig[ind + 1] = {}
    m_layer = NNparams['m' + str(ind+1)]
    b_layer = NNparams['b' + str(ind+1)]
    
    for i in range(m_layer.shape[0]):
        # get weights and biases
        m = m_layer[i]; b = b_layer[i]
        split = getSplit(m, nPartitions)
        
        # compute upper and lower bounds for full output
        ub = sum(x[ind][j].UB*max(0, m[j])+ x[ind][j].LB*min(0, m[j]) for j in range(len(m))) + b
        lb = sum(x[ind][j].UB*min(0, m[j])+ x[ind][j].LB*max(0, m[j]) for j in range(len(m))) + b

        if ind == nLayers - 1:
            # No ReLU in final layer
            x[ind + 1][i] = model.addVar(lb, ub, name='x_' + str(ind + 1) + '_' + str(i))
            model.addConstr(x[ind + 1][i] == sum(x[ind][j] * m[j] for j in range(len(m))) + b)

        else:
            # define vars
            x[ind + 1][i] = model.addVar(0, max(0, ub), name='x_' + str(ind + 1) + '_' + str(i))
            sig[ind + 1][i] = model.addVar(0, 1, vtype=gb.GRB.BINARY, name='sigma_' + str(ind + 1) + '_' + str(i))
            z2[ind + 1][i] = {}

            for j in range(len(split)):
                # compute upper and lower bounds for split
                ub = sum(x[ind][n].UB * max(0, m[n]) + x[ind][n].LB * min(0, m[n]) for n in split[j])
                lb = sum(x[ind][n].UB * min(0, m[n]) + x[ind][n].LB * max(0, m[n]) for n in split[j])

                # create auxiliary variables
                z2[ind + 1][i][j] = model.addVar(min(0, lb), max(0, ub), name='z2_' + str(ind + 1) + '_' + str(i) + '_' + str(j))

                # auxiliary variable bounds
                model.addConstr(sum(x[ind][n] * m[n] for n in split[j]) - z2[ind + 1][i][j] >= sig[ind + 1][i] * lb)
                model.addConstr(sum(x[ind][n] * m[n] for n in split[j]) - z2[ind + 1][i][j] <= sig[ind + 1][i] * ub)
                model.addConstr(z2[ind + 1][i][j] >= (1 - sig[ind + 1][i]) * lb)
                model.addConstr(z2[ind + 1][i][j] <= (1 - sig[ind + 1][i]) * ub)

            # partial hull representation of node i
            model.addConstr(x[ind + 1][i] == sum(z2[ind + 1][i][j] for j in range(len(z2[ind + 1][i]))) + b * (1 - sig[ind + 1][i]))
            model.addConstr(sum(sum(x[ind][n] * m[n] for n in split[j]) - z2[ind + 1][i][j] for j in range(len(z2[ind + 1][i]))) +
                            sig[ind + 1][i] * b <= 0)
            model.addConstr(sum(z2[ind + 1][i][j] for j in range(len(z2[ind + 1][i]))) + (1 - sig[ind + 1][i]) * b >= 0)

            ub_tot = sum(x[ind][j].UB*max(0, m[j])+ x[ind][j].LB*min(0, m[j]) for j in range(len(m))) 
            lb_tot = sum(x[ind][j].UB*min(0, m[j])+ x[ind][j].LB*max(0, m[j]) for j in range(len(m))) 
            model.addConstr(sum(z2[ind + 1][i][j] for j in range(len(z2[ind + 1][i]))) <= ub_tot * (1 - sig[ind + 1][i]))
            model.addConstr(sum(z2[ind + 1][i][j] for j in range(len(z2[ind + 1][i]))) >= lb_tot * (1 - sig[ind + 1][i]))
            model.addConstr(sum(x[ind][n] * m[n] for n in range(len(x[ind]))) - sum(z2[ind + 1][i][j] for j in range(len(
                    z2[ind + 1][i]))) >= sig[ind + 1][i] * lb_tot)
            model.addConstr(sum(x[ind][n] * m[n] for n in range(len(x[ind]))) - sum(z2[ind + 1][i][j] for j in range(len(
                        z2[ind + 1][i]))) <= sig[ind + 1][i] * ub_tot)

    model.update()

#OPTIMIZE MODEL
model.setObjective(-(x[ind + 1][targetClass]))
model.setParam('MIPFocus',3)
model.setParam('Cuts',1)
model.setParam('PumpPasses', 100)
model.setParam('Method', 1)
model.setParam('TimeLimit',1800)
           
model.optimize()

        
      
