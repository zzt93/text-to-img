import torch

# 4 y, 3 dim
y_f = torch.tensor([[1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0]])
# 4 samples, top3
topk_index = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1], [3, 2, 1, 3, 2, 3, 2, 2]])

nm = y_f
# 向量标准化，向量模为1，计算cos时就不需要除了
nm /= torch.norm(nm, dim=1).view(-1, 1)
# 向量点积
cosine = torch.sum(nm[topk_index[0, :], :] * nm[topk_index[1, :], :], 1)
# math.min(1, cosine)
cosine = torch.min(torch.ones([cosine.shape[0]]), cosine)
# 面之间的弧度
theta = torch.acos(cosine)
print('OT 角度 {}'.format(theta * 180 / torch.pi))

condition = theta < 1
indices = torch.nonzero(condition).squeeze()