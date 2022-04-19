import sys
sys.path.append('../')
import math


dataRead=open("./data/tox21.csv","r")#temporarily just using the same dataset for calibration
data=dataRead.readlines()
data=data[1:]
data=[x.strip().split(",") for x in data]


data=[x[1:] for x in data]#truncating header and smile names, and just use integer indexing

N=len(data)#number of points in the dataset!
K=len(data[0])#number of classes
classData=[0]*N#classData[i]=Y_i

for x in range(N):
    for y in range(K):
        #print((x,y))
        if data[x][y]=="":
            data[x][y]=0
        else:
            data[x][y]=int(data[x][y])
            if data[x][y]==1:
                classData[x]=y

calibrationSet=[]#just to make it line up with literature a bit more
for i in range(len(classData)):
    calibrationSet.append((i,classData[i]))#X,Y pair


scoreRead=open("tox21_preds.csv","r")#loading from csv predictor rather than running the model every time
score=scoreRead.readlines()
score=score[1:]
score=[x.strip().split(",") for x in score]
score=[[float(_) for _ in x[1:]] for x in score]#truncating header and smile names!!


##############################################################################
##############################################################################


def sBasic(x,y,score,K):#x,y are the indices
        return -score[x][y]

def sAdaptive(x,y,score,K):#x,y are the indices
    result=0
    for i in range(K):
        if score[x][i]>=score[x][y]:
            result+=score[x][i]
    return result

#TODO: Change N,K inputs to lists of input data and classes
#TODO: Change to using pandas?

def calculateQhat(calibrationSet,score,s,alpha,N,K):

    #Right now treating score as a predetermined matrix. In reality it would be the results of our trained model fhat

    
    calibrationScores=[]

    for (X,Y) in calibrationSet:
        calibrationScores.append(s(X,Y,score,K))

    calibrationScores.sort()
    index=int(math.ceil((1-alpha)*(N+1)))-1
    qhat=calibrationScores[index]#want the ceil((1-alpha)(N+1))th value

    return qhat

def returnSet(X,qhat,s,score,K):
    set=[]
    for i in range(K):
        if s(X,i,score,K)>=qhat:
            set.append(i)

    return set

alpha=0.2
s=sBasic
qhat=calculateQhat(calibrationSet,score,s,alpha,N,K)

for i in range(100):
    X=i
    print(returnSet(X,qhat,s,score,K))
    print(len(returnSet(X,qhat,s,score,K)))



    
    