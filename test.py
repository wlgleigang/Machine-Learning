from torch.autograd import Variable
import torch
import torch.autograd
x=torch.randn(3)
x=Variable(x,requires_grad=True)
y=x*2
print(y)
y.backward(torch.FloatTensor([1,0.1,0.01]))
print(x.grad)


