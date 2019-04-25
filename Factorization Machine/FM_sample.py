import numpy as np


def initialize_v(n, k):
    v=np.mat(0.2*np.random.randn(n,k))
    return v


def getPrediction(dataMatrix, w0, w, v):
    m=np.shape(dataMatrix)[0]
    result=[]
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2
        p = w0 + dataMatrix[x] * w + interaction
        pre=sigmoid(p[0,0])
        result.append(pre)
    return result
def getAccuracy(predict,classLabels):
    m=len(predict)
    allItem=0
    error=0
    for i in range(m):
        allItem+=1
        if float(predict[i])<0.5 and classLabels[i]==1.0:
            error+=1
        elif float(predict[i])>=0.5 and classLabels[i]==-1.0:
            error+=1
        else:
            continue
    return float(error)/allItem



def getCost(predict, classLabels):
    m=len(predict)
    error=0.0
    for i in range(m):
        error-=np.log(sigmoid(predict[i]*classLabels[i]))
    return error


def stocGradAscent(dataMatrix,classLabels,k,max_iter,alpha):
    m,n=np.shape(dataMatrix)
    w=np.zeros((n,1))
    w0=0
    v=initialize_v(n,k)
    for it in range(max_iter):
        print("iteration:",it)
        for x in range(m):
            inter_1=dataMatrix[x]*v
            inter_2=np.multiply(dataMatrix[x],dataMatrix[x])*np.multiply(v,v)
            interaction=np.sum(np.multiply(inter_1,inter_1)-inter_2)/2
            p=w0+dataMatrix[x]*w+interaction
            loss=sigmoid(classLabels[x]*p[0,0])-1
            w0=w0-alpha*loss*classLabels[x]
            for i in range(n):
                if dataMatrix[x,i]!=0:
                    w[i,0]=w[i,0]-alpha*loss*classLabels[x]*dataMatrix[x,i]
                    for j in range(k):
                        v[i,j]=v[i,j]-alpha*loss*classLabels[x]*(dataMatrix[x,i]*inter_1[0,j]-v[i,j]*dataMatrix[x,i]*dataMatrix[x,i])
        if it%1000==0:
            print("\t------iter:",it,",cost:",getCost(getPrediction(np.mat(dataTrain),w0,w,v),classLabels))
    return w0,w,v
def sigmoid(inx):
    return 1.0/(1.0+np.exp(-inx))


def loadDataSet(data):
    dataMat=[]
    labelMat=[]
    fr=open(data)
    for line in fr.readlines():
        lines=line.strip().split("\t")
        lineArr=[]
        for i in range(len(lines)-1):
            lineArr.append(float(lines[i]))
        dataMat.append(lineArr)
        labelMat.append(float(lines[-1])*2-1)
    fr.close()
    return dataMat,labelMat



def save_model(filename, w0, w, v):
    f=open(filename,"w")
    f.write(str(w0)+"\n")
    w_array=[]
    m=np.shape(w)[0]
    for i in range(m):
        w_array.append(str(w[i,0]))
    f.write("\t".join(w_array)+"\n")
    m1,n1=np.shape(v)
    for i in range(m1):
        v_tmp=[]
        for j in range(n1):
            v_tmp.append(str(v[i,j]))
        f.write("\t".join(v_tmp)+"\n")
    f.close()

if __name__=="__main__":
    print("---------1.load data----------")
    dataTrain,labelTrain=loadDataSet("D:\pychar_projects\dnc-master\ML02\Factorization Machine\data.txt")
    print(np.array(dataTrain).shape)
    print("---------2.learning----------")
    w0,w,v=stocGradAscent(np.mat(dataTrain),labelTrain,3,10000,0.01)
    print(w0,w,v,sep="\n")
    predict_result=getPrediction(np.mat(dataTrain),w0,w,v)
    print("---------training accuracy:%f" %(1-getAccuracy(predict_result,labelTrain)))
    print("---------3.save model---------")
    save_model("weights",w0,w,v)






