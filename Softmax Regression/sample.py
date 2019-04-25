import numpy as np

def gradientAscent(feature_data,label_data,k,maxCycle,alpha):
    m,n=np.shape(feature_data)
    weights=np.mat(np.ones((n,k)))
    i=0
    while i<=maxCycle:
        err=np.exp(feature_data*weights)#m,k
        if i%100 ==0:
            print("\t-----iter:",i,",cost: ",cost(err,label_data))
        rowsum=-err.sum(axis=1)
        rowsum=rowsum.repeat(k,axis=1)
        err=err/rowsum
        for x in range(m):
            err[x,label_data[x,0]]+=1
        weights=weights+(alpha/m)*feature_data.T*err
        i+=1
    return weights
def cost(err,label_data):
    m=np.shape(err)[0]
    sum_cost=0.0
    for i in range(m):
        if err[i,label_data[i,0]]/np.sum(err[i,:])>0:
            sum_cost-=np.log(err[i,label_data[i,0]]/np.sum(err[i,:]))
        else:
            sum_cost-=0
    return sum_cost/m
def load_data(input_file):
    f=open(inputfile)
    feature_data=[]
    label_data=[]
    for line in f.readlines():
        feature_tmp=[]
        feature_tmp.append(1.0)
        lines=line.strip().split("\t")
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_data.append(int(lines[-1]))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data),np.mat(label_data).T,len(set(label_data))
def save_model(filename,weights):
    f_w=open(filename,"w")
    m,n=np.shape(weights)
    for i in range(m):
        w_tmp=[]
        for j in range(n):
            w_tmp.append(str(weights[i,j]))
            f_w.write("\t".join(w_tmp)+"\n")
    f_w.close()
if __name__=="__main__":
    inputfile="D:\pychar_projects\dnc-master\ML02\Softmax Regression\SoftInput.txt"
    print("----------1.load data-----------")
    feature,label,k=load_data(inputfile)
    print("----------2.training------------")
    weights=gradientAscent(feature,label,k,10000,0.4)
    print("----------3.save model----------")
    save_model("weights",weights)
