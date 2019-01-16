import torch
from torch import optim
from net.hourglass import CornerNet
from loss.loss import AELoss
from data_set.data_gen import get_batch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
net = CornerNet()
loss =  AELoss(pull_weight=1e-1, push_weight=1e-1)
net.cuda()
optimer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00005)

def gen_tensor(x):
    x =  torch.autograd.Variable(torch.from_numpy(x).cuda())
    return x



def train():
    net.train()
    gen = get_batch(batch_size=2, class_name='voc', max_detect=100)
    for step in range(1000):
        data = next(gen)
        data = [gen_tensor(x) for x in data]
        imges = data[0]
        tl_ind = data[1]
        br_ind = data[2]
        output = net(imges, tl_ind, br_ind)
        ls = loss(output, data[3:])
        ls = ls.mean()
        optimer.zero_grad()
        ls.backward()
        optimer.step()
        print(ls.item())
    torch.save(net.state_dict(),'net.pth')



for epoch in range(100):
    train()