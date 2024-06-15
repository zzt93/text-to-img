import numpy as np

from pyOMT_raw import OMTRaw, train_omt
import torch
import scipy.io as sio


def compute_ot(feature_input_path, ot_model_path, gen_feature_path, train: bool = True, thresh=0.7, topk=20, dissim=0.75, max_gen_samples=None):
    """args for omt"""
    h_P = torch.load(feature_input_path)
    num_P = h_P.shape[0]
    dim_y = h_P.shape[1]
    max_iter = 20000
    lr = 5e-2
    bat_size_P = num_P
    bat_size_n = 1000
    init_num_bat_n = 20
    if not train:
        max_iter = 0

    # crop h_P to fit bat_size_P
    h_P = h_P[0 : num_P//bat_size_P*bat_size_P, :]
    num_P = h_P.shape[0]

    p_s = OMTRaw(h_P, num_P, dim_y, max_iter, lr, bat_size_P, bat_size_n)
    '''train omt'''
    if train:
        train_omt(p_s, init_num_bat_n)
        torch.save(p_s.d_h, ot_model_path)
    else:
        num_gen_x = 20 * bat_size_n
        p_s.set_h(torch.load(ot_model_path))
        '''generate new samples'''
        gen_P(p_s, num_gen_x, gen_feature_path, thresh=thresh, topk=topk, dissim=dissim, max_gen_samples=max_gen_samples)


def gen_P(p_s, numX, output_P_gen, thresh=-1, topk=5, dissim=0.75, max_gen_samples=None):
    I_all = -torch.ones([topk, numX], dtype=torch.long)
    num_bat_x = numX // p_s.bat_size_n
    bat_size_x = min(numX, p_s.bat_size_n)
    for ii in range(max(num_bat_x, 1)):
        p_s.pre_cal(ii)
        p_s.cal_measure()
        _, I = torch.topk(p_s.d_U, topk, dim=0)
        for k in range(topk):
            I_all[k, ii * bat_size_x:(ii + 1) * bat_size_x].copy_(I[k, 0:bat_size_x])
    I_all_2 = -torch.ones([2, (topk - 1) * numX], dtype=torch.long)
    for ii in range(topk - 1):
        I_all_2[0, ii * numX:(ii + 1) * numX] = I_all[0, :]
        I_all_2[1, ii * numX:(ii + 1) * numX] = I_all[ii + 1, :]
    I_all = I_all_2

    if torch.sum(I_all < 0) > 0:
        print('Error: numX is not a multiple of bat_size_n')

    '''compute angles'''
    P = p_s.h_P
    nm = torch.cat([P, -torch.ones(p_s.num_P, 1)], dim=1)
    nm /= torch.norm(nm, dim=1).view(-1, 1)
    cs = torch.sum(nm[I_all[0, :], :] * nm[I_all[1, :], :], 1)  # element-wise multiplication
    cs = torch.min(torch.ones([cs.shape[0]]), cs)
    theta = torch.acos(cs)
    # pdb.set_trace()

    '''filter out generated samples with theta larger than threshold'''
    I_gen = I_all[:, theta <= thresh]
    I_gen, _ = torch.sort(I_gen, dim=0)
    # _, uni_gen_id = np.unique(I_gen.numpy(), return_index=True, axis=1)
    _, uni_gen_id = np.unique(I_gen[0, :].numpy(), return_index=True)
    np.random.shuffle(uni_gen_id)
    I_gen = I_gen[:, torch.from_numpy(uni_gen_id)]
    # pdb.set_trace()

    numGen = I_gen.shape[1]
    if max_gen_samples is not None:
        numGen = min(numGen, max_gen_samples)
    I_gen = I_gen[:, :numGen]
    print('OT successfully generated {} samples'.format(
        numGen))

    '''generate new features'''
    # rand_w = torch.rand([numGen,1])
    rand_w = dissim * torch.ones([numGen, 1])
    P_gen = (torch.mul(P[I_gen[0, :], :], 1 - rand_w) + torch.mul(P[I_gen[1, :], :], rand_w)).numpy()

    P_gen2 = P[I_gen[0, :], :]
    P_gen = np.concatenate((P_gen, P_gen2))

    id_gen = I_gen[0, :].squeeze().numpy().astype(int)

    sio.savemat(output_P_gen, {'features': P_gen, 'ids': id_gen})
