import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def load_data(filename):
    f=open(filename)
    features=[]
    labels=[]
    for line in f.readlines():
        feature_tmp=[]
        label_tmp=[]
        lines=line.strip().split("\t")
        feature_tmp.append(1.0)
        for i in np.arange(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))

        features.append(feature_tmp)
        labels.append(label_tmp)
    f.close()
    return np.mat(features),np.mat(labels)

def lr_train_bgd(features,labels,maxCycle,alpha):
    n=np.shape(features)[1]
    w=np.mat(np.ones((n,1)))
    i=0
    while(i<=maxCycle):
        i+=1
        h=sigmoid(features*w)
        err=labels-h
        if i % 100==0:
            print("\t----------iter="+str(i)+"错误率:"+str(err_rate(h,labels)))
        w=w+alpha*features.T*err
    return w

def err_rate(h,labels):
    m=np.shape(h)[0]
    sum_err=0.0
    for i in np.arange(m):
        if (h[i,0]>0) and (1-h[i,0]>0):
            sum_err-=labels[i,0]*np.log(h[i,0])+(1-labels[i,0]*np.log(1-h[i,0]))
        else:
            sum_err-=0
    return sum_err/m

def save_model(filename,w):
    m=np.shape(w)[0]
    f=open(filename,'w')
    w_array=[]
    for i in np.arange(m):
        w_array.append(str(w[i,0]))
    f.write("\t".join(w_array))
    f.close()
if __name__ =="__main__":
    print("--------------1.load data--------------")
    features,labels=load_data("D:\pychar_projects\dnc-master\ML02\Logistic Regression\data.txt")
    print("--------------2.training---------------")
    w=lr_train_bgd(features,labels,1000,0.01)
    print("--------------3.save model-------------")
    save_model("copy01_weight",w)




