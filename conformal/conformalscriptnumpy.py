import sys
sys.path.append('../')
import math

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import numpy as np


data=pd.read_csv("./data/tox21.csv")#temporarily just using the same dataset for calibration
data=data.fillna(0)
data=data.drop(columns=['smiles'])
data=data.to_numpy()

softmax_scores=pd.read_csv("tox21_preds.csv")#temporarily just using the same dataset for calibration
softmax_scores=softmax_scores.drop(columns=['smiles'])
softmax_scores=softmax_scores.to_numpy()


(N, K) = data.shape
true_class = np.zeros(N, dtype=np.int32)
calibration_set = np.zeros((N,2), dtype=np.int32)

for x in range(N):
    for y in range(K):
        if data[x][y] == 1:
            true_class[x] = y
            calibration_set[x][0] = x
            calibration_set[x][1] = y



def s_basic(x,y,softmax_scores,K):#x,y are the indices
        return -softmax_scores[x][y]

def s_adaptive(x,y,softmax_scores,K):#x,y are the indices
    result = 0
    for Y in range(K):
        if softmax_scores[x][Y] >= softmax_scores[x][y]:
            result += softmax_scores[x][Y]
    return result


def calculate_qhat(calibration_set,softmax_scores,s,alpha,N,K):

    #Right now treating score as a predetermined matrix. In reality it would be the results of our trained model fhat

    calibration_scores=np.zeros(N+1)

    for x in range(N):
        X, Y = calibration_set[x][0], calibration_set[x][1]
        calibration_scores[x]=s(X,Y,softmax_scores,K)

    calibration_scores[N]=np.Inf

    calibration_scores=np.sort(calibration_scores)
    index=int(math.ceil((1-alpha)*(N+1)))-1
    qhat=calibration_scores[index]#want the ceil((1-alpha)(N+1))th value

    return qhat

def returnSet(X,qhat,s,softmax_scores,K):
    set=np.zeros(K, dtype=np.int32)
    for i in range(K):
        if s(X,i,softmax_scores,K)>=qhat:
            set[i]=1

    return set

alpha = 0.5
s = s_basic
qhat = calculate_qhat(calibration_set,softmax_scores,s,alpha,N,K)

set_sizes = np.zeros(N, dtype=np.int32)
ssc_proportion = np.zeros(K+1)
ssc_total = np.zeros(K+1)


for X in range(N):
    #print(returnSet(X,qhat,s,score,K))
    x_set = returnSet(X,qhat,s,softmax_scores,K)
    set_sizes[X] = sum(x_set)
    ssc_total[sum(x_set)] += 1
    if x_set[true_class[X]] == 1:
        ssc_proportion[sum(x_set)]+=1

for i in range(K+1):
    if ssc_proportion[i]>0:
        ssc_proportion[i]=ssc_proportion[i]/ssc_total[i]

print(ssc_proportion)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

plt.hist(set_sizes, density=True, bins=12)
plt.show()









    
    