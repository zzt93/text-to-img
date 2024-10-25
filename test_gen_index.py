import torch

max_gen_samples = 10
dissim = 0.7
sample_size = 4
samples = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        ])
y_features = torch.tensor([[-0.5318,  0.1094, -1.1009, -0.5525],
        [ 0.0938, -0.4701,  1.3861, -1.1568],
        [-0.5866, -0.7426, -0.2033, -0.6423],
        [-0.2315, -0.9048, -1.1729,  1.5507],
        [ 0.3560,  0.9750, -0.0887, -1.9549],
        [ 0.1260, -0.8927, -1.8934,  0.4291],
        [ 1.6381,  0.3503, -1.9301, -0.2514],
        [-0.1265,  2.7628, -1.1591, -0.1178],
        [-0.2116,  1.4087,  0.8956, -1.2394],
        [-0.8895,  0.4530, -2.0512,  0.9730],
        [-0.7549,  0.3797, -0.8166,  1.9542],
        [-1.1226,  0.4356, -1.3921,  0.0991]])
# 4 samples, top3, 12 yi(0~11)
# topk_index = torch.tensor([[7, 11, 10, 9, 7, 11, 10, 9], [3, 2, 1, 3, 2, 3, 2, 2]])
topk_index = torch.tensor([[7, 11, 7, 9, 7, 11, 7, 9],
                           [3, 2, 1, 3, 2, 3, 5, 6]])


theta = torch.tensor([0.7854, 0.7854, 1.5708, 1.5708, 1.0472, 1.5708, 1.0472, 0.7854])
thresh = 1.1  # threshold value

condition = theta <= thresh
_ = topk_index[:, condition]  # Filter columns (pick columns where theta <= 0.3)
# Result: gen_index is:
# tensor([[ 7, 11,  7,  7,  9],
#         [ 3,  2,  2,  5,  6]])
indices = torch.nonzero(condition).squeeze()
# tensor([0, 1, 4, 6, 7])

seen_mod_results = {}
unique_indices = []

# Iterate through the tensor
for i, value in enumerate(indices):
    mod_result = value.item() % sample_size
    if mod_result not in seen_mod_results:
        seen_mod_results[mod_result] = True
        unique_indices.append(value.item())
# [0 1 6 7]
gen_index = topk_index[:, unique_indices]


num_gen = gen_index.shape[1]
if max_gen_samples is not None:
    num_gen = min(num_gen, max_gen_samples)
gen_index = gen_index[:, :num_gen]

rand_w = dissim * torch.ones([num_gen, 1])
# 加权平均
gen_feature = (torch.mul(y_features[gen_index[0, :], :], 1 - rand_w) + torch.mul(y_features[gen_index[1, :], :],
                                                                                 rand_w)).numpy()
