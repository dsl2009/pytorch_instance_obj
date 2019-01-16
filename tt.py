import torch
a = torch.tensor([1,2,3])

b = torch.randn(2,4)
d = torch.randn(2,1)




print(b[0]/d[0])
print(b[1]/d[1])

print(b/d)