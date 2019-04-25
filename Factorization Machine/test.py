import numpy as np

def loadDataSet(data):
    dataMat=[]
    fr=open(data)
    for line in fr.readlines():
        lines=line.strip().split("\t")
        lineArr=[]
        for i in range(len(lines)):
            lineArr.append(float(lines[i]))
        dataMat.append(lineArr)
    fr.close()
    return dataMat
def loadModel(model_file):
    f=open(model_file)
    line_index=0
    w0=0.0
    w=[]
    v=[]
    for line in f.readlines():
        lines=line.strip().split("\t")
        if line_index==0:
            w0=float(lines[0].strip())
        elif line_index==1:
            for x in lines:
                w.append(float(x.strip()))
        else:
            v_tmp=[]
            for x in lines:
                v_tmp.append(float(x.strip()))
            v.append(v_tmp)
        line_index+=1
    f.close()
    return w0,np.mat(w).T,np.mat(v)
def save_result(file_name,result):
    f=open(file_name,"w")
    f.write("\n".join(str(x) for x in result))
    f.close()
def sigmoid(inx):
    return 1.0/(1.0+np.exp(-inx))
def getPrediction(dataMatrix, w0, w, v):
    m=np.shape(dataMatrix)[0]
    result=[]
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2
        print(dataMatrix[x])
        p = w0 + dataMatrix[x] * w + interaction
        pre=sigmoid(p[0,0])
        result.append(pre)
    return result
if __name__=="__main__":
    dataTest=loadDataSet("D:\pychar_projects\dnc-master\ML02\Factorization Machine\\test_data.txt")
    print(np.array(dataTest).shape)
    w0,w,v=loadModel("weights")
    print(w)
    result=getPrediction(dataTest,w0,w,v)
    save_result("predict_result",result)
