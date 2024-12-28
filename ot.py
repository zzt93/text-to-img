import numpy as np

import config
import util
from pyOMT_raw import OMTRaw, train_omt
import torch
import scipy.io as sio


def compute_ot(cpu_features: torch.tensor, ot_model_dir, ot_opt: dict):
    print('compute_ot [ot_opt={}]'.format(ot_opt))
    points_num = cpu_features.shape[0]
    dim_y = cpu_features.shape[1]
    bat_size_device = points_num

    # crop cpu_features to fit bat_size_device
    cpu_features = cpu_features[0 : points_num//bat_size_device*bat_size_device, :]
    points_num = cpu_features.shape[0]

    ot = OMTRaw(cpu_features, points_num, bat_size_device, dim_y, **ot_opt, model_dir=ot_model_dir)

    init_sample_batch = 20
    train_omt(ot, init_sample_batch)
    torch.save(ot.d_h, ot.h_path())


def generate_sample_and_ot(cpu_features: torch.tensor, ot_model_dir, gen_feature_path, ot_opt: dict, thresh=0.7, topk=20, dissim=0.75, max_gen_samples=100, sample_batch=20, **kwargs):
    points_num = cpu_features.shape[0]
    dim_y = cpu_features.shape[1]
    ot = OMTRaw(cpu_features, points_num, points_num, dim_y, **ot_opt, model_dir=ot_model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ot.set_h(torch.load(ot.h_path(), map_location=torch.device(device=device)))
    gen_point_and_ot(ot, sample_batch, gen_feature_path, thresh=thresh, topk=topk, dissim=dissim, max_gen_samples=max_gen_samples)


def ot_a_sample(p_s: OMTRaw, sample: torch.Tensor, thresh: float = 0.35, topk: int = 4, dissim: float = 0.75, **kwargs) -> torch.Tensor:
    return gen_point_and_ot(p_s, 1, '', thresh=thresh, topk=topk, dissim=dissim, max_gen_samples=1, provided_sample=sample.view(1, -1))


def gen_point_and_ot(p_s: OMTRaw, sample_batch, generate_path, thresh: float, topk: int, dissim: float, max_gen_samples: int, provided_sample: torch.Tensor = None) -> torch.Tensor:
    print('generate feature [sample_batch={}, generate_path={}, thresh={}, topk={}]'.format(sample_batch, generate_path, thresh, topk))
    num_x = p_s.count_of_x_in_batch * sample_batch
    bat_size_x = p_s.count_of_x_in_batch

    topk_index = -torch.ones([topk, num_x], dtype=torch.long)
    samples = torch.empty((num_x, p_s.dim), dtype=torch.float)
    for ii in range(sample_batch):
        if provided_sample is None:
            p_s.generate_samples(ii)
        else:
            p_s.d_sampled_x = provided_sample
        samples[ii * bat_size_x:(ii + 1) * bat_size_x, :] = p_s.d_sampled_x
        p_s.calculate_energy_for_sampled_x()
        # uₕ(x).shape = [bat_y, bat_x]： 每一个x带入每个uₕ(x)函数（超平面函数），得到的的值
        # 计算每个sample，对应的前topk大的超平面的索引（即对应的yi的index）
        _, point_index = torch.topk(p_s.U_h_x, topk, dim=0)
        for k in range(topk):
            topk_index[k, ii * bat_size_x:(ii + 1) * bat_size_x].copy_(point_index[k, 0:bat_size_x])


    # [topk, num_x] 铺平成 [2, (topk - 1) * num_x]
    # [原来第0行, 原来第0行 ... 原来第0行]
    # [原来第1行, 原来第2行 ... 原来第topk-1行]
    topk_index = reshape(topk_index, num_x, topk)

    if torch.sum(topk_index < 0) > 0:
        print('{}'.format(topk_index))

    '''compute angles'''
    # yi 是面的法向量，可以直接用于计算面的夹角
    y_features = p_s.y_features
    # ??? https://github.com/k2cu8/pyOMT/issues/4
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    nm = torch.cat([y_features, -torch.ones(p_s.y_nums, 1).to(device)], dim=1)
    # nm = y_features
    # 向量标准化，向量模为1，计算cos时就不需要除了
    nm /= torch.norm(nm, dim=1).view(-1, 1)
    # 向量点积
    cosine = torch.sum(nm[topk_index[0, :], :] * nm[topk_index[1, :], :], 1)
    # math.min(1, cosine)
    cosine = torch.min(torch.ones([cosine.shape[0]]).to(device), cosine)
    # 面之间的弧度(top0&top1 * num_x, top0&top2 * num_x, ... top0&topk-1 * num_x)
    # shape: [(topk - 1) * num_x]
    theta = torch.acos(cosine)
    print('OT 弧度 {}'.format(theta))
    print('OT 角度 {}'.format(theta * 180 / torch.pi))
    # pdb.set_trace()

    '''filter out generated samples with theta larger than threshold angle'''
    # gen_index = topk_index[:, theta <= thresh]
    # why sort???
    # gen_index, _ = torch.sort(gen_index, dim=0)
    # _, uni_gen_id = np.unique(gen_index.numpy(), return_index=True, axis=1)
    # _, uni_gen_id = np.unique(gen_index[0, :].numpy(), return_index=True)
    # np.random.shuffle(uni_gen_id)
    # gen_index = gen_index[:, torch.from_numpy(uni_gen_id)]
    # pdb.set_trace()
    condition = theta <= thresh
    _ = topk_index[:, condition]  # Filter columns (pick columns where theta <= 0.3)
    # Result: gen_index is:
    # tensor([[ 7, 11,  7,  7,  9],
    #         [ 3,  2,  2,  5,  6]])
    indices = torch.nonzero(condition).view(-1)
    if indices.numel() == 0:
        return None
    # tensor([0, 1, 4, 6, 7])

    seen_mod_results = {}
    unique_indices = []

    # Iterate through the tensor
    for i, value in enumerate(indices):
        mod_result = value.item() % num_x
        if mod_result not in seen_mod_results:
            seen_mod_results[mod_result] = True
            unique_indices.append(value.item())
    # [0 1 6 7]
    print('可以用来生成新样本的sample索引 {} '.format(unique_indices))
    gen_index = topk_index[:, unique_indices]

    num_gen = gen_index.shape[1]
    if max_gen_samples is not None:
        num_gen = min(num_gen, max_gen_samples)
    gen_index = gen_index[:, :num_gen]
    print('OT successfully generated {} samples'.format(num_gen))
    print('OT gen_index {}'.format(gen_index))

    '''generate new features'''
    # rand_w = torch.rand([num_gen,1])
    rand_w = dissim * torch.ones([num_gen, 1]).to(device)
    # 加权平均
    gen_feature = y_features[gen_index[0, :], :] * (1 - rand_w) + y_features[gen_index[1, :], :] * rand_w

    # 映射给定sample，不需要保存，直接返回
    if provided_sample is not None:
        return gen_feature

    # include directly mapped feature
    # P_gen2 = y_features[gen_index[0, :], :]
    # gen_feature = np.concatenate((gen_feature, P_gen2))
    generate_transformer_data(gen_index, unique_indices, samples)

    id_gen = gen_index[0, :].squeeze().numpy().astype(int)

    sio.savemat(generate_path, {'features': gen_feature.numpy(), 'ids': id_gen})


def generate_transformer_data(y_index, x_index, x_feature):
    labels = torch.load(config.path(config.PathType.result, "./coder", "label.pth"))

    # 数字1(第几个y)x的坐标
    fmt = "数字%d(%s) %s"
    top0 = y_index[0, :]
    num_gen = top0.shape[0]
    x_num = x_feature.shape[0]
    res = []
    for i in range(min(num_gen, len(x_index))):
        # 详见 @reshape
        x = x_index[i] % x_num
        res.append(fmt % (labels[top0[i]], top0[i].item(), x_feature[x].numpy().tolist()))
    # print(res)
    p = config.path(config.PathType.train, "./transformer", config.transformer_train_data_file)
    util.save_data(p, res, "")


def reshape(topk_index, num_x, topk):
    """
    # [topk, num_x] 铺平成 [2, (topk - 1) * num_x]
    # [原来第0行, 原来第0行 ... 原来第0行]
    # [原来第1行, 原来第2行 ... 原来第topk-1行]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    tmp = -torch.ones([2, (topk - 1) * num_x], dtype=torch.long).to(device)
    for ii in range(topk - 1):
        tmp[0, ii * num_x:(ii + 1) * num_x] = topk_index[0, :]
        tmp[1, ii * num_x:(ii + 1) * num_x] = topk_index[ii + 1, :]
    return tmp
