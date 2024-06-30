import numpy as np

from pyOMT_raw import OMTRaw, train_omt
import torch
import scipy.io as sio


def compute_ot(cpu_features: torch.tensor, ot_model_path):
    points_num = cpu_features.shape[0]
    dim_y = cpu_features.shape[1]
    max_iter = 3000
    lr = 1e-5
    bat_size_device = points_num
    sample_batch_size = 1000
    init_num_bat_n = 20

    # crop cpu_features to fit bat_size_device
    cpu_features = cpu_features[0 : points_num//bat_size_device*bat_size_device, :]
    points_num = cpu_features.shape[0]

    ot = OMTRaw(cpu_features, points_num, dim_y, max_iter, lr, bat_size_device, sample_batch_size, model_path=ot_model_path)

    train_omt(ot, init_num_bat_n)
    torch.save(ot.d_h, ot.h_path())


def ot_map(cpu_features: torch.tensor, ot_model_path, gen_feature_path, thresh=0.7, topk=20, dissim=0.75, max_gen_samples=None, sample_batch=20):
    points_num = cpu_features.shape[0]
    dim_y = cpu_features.shape[1]
    ot = OMTRaw(cpu_features, points_num, dim_y, 0, 0, points_num, model_path=ot_model_path)
    ot.set_h(torch.load(ot.h_path()))
    gen_P(ot, sample_batch, gen_feature_path, thresh=thresh, topk=topk, dissim=dissim, max_gen_samples=max_gen_samples)


def gen_P(p_s: OMTRaw, num_bat_x, generate_path, thresh:float=-1.0, topk=5, dissim=0.75, max_gen_samples=None):
    num_x = p_s.bat_size_x * num_bat_x
    bat_size_x = p_s.bat_size_x

    topk_index = -torch.ones([topk, num_x], dtype=torch.long)
    for ii in range(max(num_bat_x, 1)):
        p_s.generate_samples(ii)
        p_s.calculate_energy_for_sampled_x()
        _, point_index = torch.topk(p_s.d_U, topk, dim=0)
        for k in range(topk):
            topk_index[k, ii * bat_size_x:(ii + 1) * bat_size_x].copy_(point_index[k, 0:bat_size_x])
    # [topk, num_x] 铺平成 [2, (topk - 1) * num_x]
    # [0_num_x, 0_num_x ... 0_num_x]
    # [1_num_x, 2_num_x ... (topk - 1)_num_x]
    topk_index = reshape(topk_index, num_x, topk)

    if torch.sum(topk_index < 0) > 0:
        print('{}'.format(topk_index))

    '''compute angles'''
    y_features = p_s.y_features_cpu
    nm = torch.cat([y_features, -torch.ones(p_s.y_nums, 1)], dim=1)
    # Normalize nm by dividing each element by the row's L2 norm:
    nm /= torch.norm(nm, dim=1).view(-1, 1)
    # element-wise multiplication, top1 * topi: dot product between top1 and other
    cs = torch.sum(nm[topk_index[0, :], :] * nm[topk_index[1, :], :], 1)
    # less the dot product, larger the angle
    cs = torch.min(torch.ones([cs.shape[0]]), cs)
    theta = torch.acos(cs)
    # pdb.set_trace()

    '''filter out generated samples with theta larger than threshold'''
    gen_index = topk_index[:, theta <= thresh]
    gen_index, _ = torch.sort(gen_index, dim=0)
    # _, uni_gen_id = np.unique(gen_index.numpy(), return_index=True, axis=1)
    _, uni_gen_id = np.unique(gen_index[0, :].numpy(), return_index=True)
    np.random.shuffle(uni_gen_id)
    gen_index = gen_index[:, torch.from_numpy(uni_gen_id)]
    # pdb.set_trace()

    num_gen = gen_index.shape[1]
    if max_gen_samples is not None:
        num_gen = min(num_gen, max_gen_samples)
    gen_index = gen_index[:, :num_gen]
    print('OT successfully generated {} samples'.format(num_gen))

    '''generate new features'''
    # rand_w = torch.rand([num_gen,1])
    rand_w = dissim * torch.ones([num_gen, 1])
    # 加权平均
    gen_feature = (torch.mul(y_features[gen_index[0, :], :], 1 - rand_w) + torch.mul(y_features[gen_index[1, :], :], rand_w)).numpy()

    P_gen2 = y_features[gen_index[0, :], :]
    # include directly mapped feature
    gen_feature = np.concatenate((gen_feature, P_gen2))

    id_gen = gen_index[0, :].squeeze().numpy().astype(int)

    sio.savemat(generate_path, {'features': gen_feature, 'ids': id_gen})


def reshape(topk_index, num_x, topk):
    """
    # [topk, num_x] 铺平成 [2, (topk - 1) * num_x]
    # [0_num_x, 0_num_x ... 0_num_x]
    # [1_num_x, 2_num_x ... (topk - 1)_num_x]
    """
    tmp = -torch.ones([2, (topk - 1) * num_x], dtype=torch.long)
    for ii in range(topk - 1):
        tmp[0, ii * num_x:(ii + 1) * num_x] = topk_index[0, :]
        tmp[1, ii * num_x:(ii + 1) * num_x] = topk_index[ii + 1, :]
    return tmp
