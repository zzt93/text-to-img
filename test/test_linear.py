import torch

b = torch.rand(30)
w = torch.rand(30,200)
q = torch.rand(15,200)

(q[-2:,:]@w.transpose(0,1))[-1,:] == q[-1,:]@w.transpose(0,1)
(q[-10:,:]@w.transpose(0,1))[-1,:] == q@w.transpose(0,1)[-1,:]