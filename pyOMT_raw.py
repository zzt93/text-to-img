import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb


class OMTRaw():
    """
    calculate mapping: x --> y

    This class is designed to compute the semi-discrete Optimal Transport (OT) problem.
    Specifically, within the unit cube [0,1]^n of the n-dim Euclidean space,
    given a source continuous distribution mu, and a discrete target distribution nu = \sum nu_i * \delta(P_i),
    where \delta(x) is the Dirac function at x \in [0,1]^n, compute the Optimal Transport map pushing forward mu to nu.

    The method is based on the variational principle of solving semi-discrete OT, (See e.g.
    Gu, Xianfeng, et al. "Variational principles for Minkowski type problems, discrete optimal transport, and discrete Monge-Ampere equations." Asian Journal of Mathematics 20.2 (2016): 383-398.)
    where a convex energy is minimized to obtain the OT map.

    Adam gradient descent method is used here to perform the optimization, and Monte-Carlo integration method is used to calculate the energy.
    """

    def __init__(self, y_features_cpu, y_nums, bat_size_y, dim, max_iter=390, lr=1e-5, count_of_x_in_batch=1000, model_dir='.', **kwargs):
        """Parameters to compute semi-discrete Optimal Transport (OT)
        Args:
            y_features_cpu: vector (CPU vector) storing locations of target points with float type and of shape (num_points, dim).
            y_nums: A positive integer indicating the number of target points (i.e. points the target discrete measure concentrates on).
            dim: A positive integer indicating the ambient dimension of OT problem.
            max_iter: A positive integer indicating the maximum steps the gradient descent would iterate.
            lr: A positive float number indicating the step length (i.e. learning rate) of the gradient descent algorithm.
            bat_size_y: Count of y in a batch of cpu_features that feeds to device (i.e. GPU). Positive integer.
            count_of_x_in_batch: Count of x in a batch of Monte-Carlo samples on device. The total number of MC samples used in each iteration is batch_size_x * num_bat.
        """
        self.model_root_path = model_dir
        self.y_features_cpu = y_features_cpu
        self.y_nums = y_nums
        self.dim = dim
        self.max_iter = max_iter
        self.lr = lr
        self.bat_size_y = bat_size_y
        self.count_of_x_in_batch = count_of_x_in_batch

        if y_nums % bat_size_y != 0:
            sys.exit('Error: (num_P) is not a multiple of (bat_size_P)')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # !internal variables
        # self.d_G_z = torch.empty(self.bat_size_x * self.dim, dtype=torch.float, device=device)
        self.d_sampled_x = torch.empty((self.count_of_x_in_batch, self.dim), dtype=torch.float, device=device)
        # self.d_h: Optimal value of h (the variable to be optimized of the variational Energy).
        self.d_h = torch.zeros(self.y_nums, dtype=torch.float, device=device)
        self.d_delta_h = torch.zeros(self.y_nums, dtype=torch.float, device=device)
        self.d_batch_index = torch.empty(self.count_of_x_in_batch, dtype=torch.long, device=device)
        self.d_batch_max = torch.empty(self.count_of_x_in_batch, dtype=torch.float, device=device)

        self.d_max_in_0_or_1 = torch.empty(self.count_of_x_in_batch, dtype=torch.long, device=device)
        self.d_total_ind = torch.empty(self.count_of_x_in_batch, dtype=torch.long, device=device)
        self.d_total_max = torch.empty(self.count_of_x_in_batch, dtype=torch.float, device=device)
        self.d_wi = torch.zeros(self.y_nums, dtype=torch.float, device=device)
        self.d_wi_sum = torch.zeros(self.y_nums, dtype=torch.float, device=device)
        self.d_adam_m = torch.zeros(self.y_nums, dtype=torch.float, device=device)
        self.d_adam_v = torch.zeros(self.y_nums, dtype=torch.float, device=device)

        # !temp variables
        # uₕ(x)
        self.U_h_x = torch.empty((self.bat_size_y, self.count_of_x_in_batch), dtype=torch.float, device=device)
        self.d_temp_h = torch.empty(self.bat_size_y, dtype=torch.float, device=device)
        self.d_temp_targets = torch.empty((self.bat_size_y, self.dim), dtype=torch.float, device=device)

        # !random number generator
        self.qrng = torch.quasirandom.SobolEngine(dimension=self.dim)

        if torch.cuda.is_available():
            print('Allocated GPU memory: {}MB'.format(torch.cuda.memory_allocated() / 1e6))
            print('Cached memory: {}MB'.format(torch.cuda.memory_cached() / 1e6))

    def generate_samples(self, count):
        """Monte-Carlo sample generator.
        Args:
            count: Index of MC mini-batch to generate in the current iteration step. Used to set the state of random number generator.
        Returns:
            self.d_volP: Generated mini-batch of MC samples on device (i.e. GPU) of shape (self.bat_size_n, dim).
        """
        # range [0, 1)
        self.qrng.draw(self.count_of_x_in_batch, out=self.d_sampled_x)
        # range [-0.5, 0.5)
        self.d_sampled_x.add_(-0.5)

    def calculate_energy_for_sampled_x(self):
        """
        计算Brenier势函数 uₕ : Ω → ℝ，uₕ(x) = maxᵢ₌₁ⁿ {πₕ,ᵢ(x)}，其中πₕ,ᵢ(x) = ⟨x, yᵢ⟩ + hᵢ是对应于yᵢ ∈ Y的支撑平面
        Wi 是源分布的胞腔分解，wi是他的度量
        ci 是 Wi的质心，也是势能平面和源分布之间的体积

        计算每个单元 \(W_i(h)\) 的 \(\mu\)-体积 \(w_i(h)\)，这可以使用传统的蒙特卡洛方法进行估计。
        我们从 \(\mu\) 分布中抽取 \(N\) 个随机样本 \(\{x_j\} \sim_{i.i.d.} \mu\)，每个单元的估计 \(\mu\)-体积为 \(\hat{w}_i(h) = \#\{j \in \mathcal{J} \mid x_j \in W_i(h)\}/N\)。
        给定 \(x_j\)，我们可以通过 \(i = \arg\max_i \{(x_j, y_i) + h_i\}, i = 1, 2, \ldots, n\) 找到 \(x_j \in W_i \)。
        当 \(N\) 足够大时，\(\hat{w}_i(h)\) 会收敛到 \(w_i(h)\)。
        """
        self.d_total_max.fill_(-1e30)
        self.d_total_ind.fill_(-1)
        i = 0
        while i < self.y_nums // self.bat_size_y:
            temp_targets = self.y_features_cpu[i * self.bat_size_y:(i + 1) * self.bat_size_y]
            temp_targets = temp_targets.view(temp_targets.shape[0], -1)

            '''U=Y@X+H'''
            self.d_temp_h = self.d_h[i * self.bat_size_y:(i + 1) * self.bat_size_y]
            self.d_temp_targets.copy_(temp_targets)
            # matrix multiple: [batch_size_y, dim] @ [batch_size_x, dim].transpose = [batch_size_y, batch_size_x]
            torch.mm(self.d_temp_targets, self.d_sampled_x.t(), out=self.U_h_x)
            torch.add(self.U_h_x, self.d_temp_h.expand([self.count_of_x_in_batch, -1]).t(), out=self.U_h_x)

            '''compute batch max'''
            # This line computes the maximum values and their indices for each X (compare across Y, dimension 0 of self.U_h_x).
            # The maximum values are stored in self.d_ind_val and the indices are stored in self.d_index.
            # [batch_size_y, batch_size_x] => batch_size_x, batch_size_x
            torch.max(self.U_h_x, 0, out=(self.d_batch_max, self.d_batch_index))
            '''update to real index in all targets'''
            self.d_batch_index.add_(i * self.bat_size_y)
            '''store max value across batches'''
            # chose between d_total_max & d_batch_max
            torch.max(torch.stack((self.d_total_max, self.d_batch_max)), 0,
                      out=(self.d_total_max, self.d_max_in_0_or_1))
            # chose between d_total_ind & d_index
            self.d_total_ind = torch.stack((self.d_total_ind, self.d_batch_index))[
                self.d_max_in_0_or_1, torch.arange(self.count_of_x_in_batch)]
            '''add step'''
            i = i + 1

        '''计算wi，从而通过梯度下降，优化h'''
        # group by Yi, then count
        self.d_wi.copy_(torch.bincount(self.d_total_ind, minlength=self.y_nums))
        # calculate ŵ i(h):每个单元的μ-体积是ŵ i(h)= #{j ∈ J | x_j ∈ W_i(h)}/N
        self.d_wi.div_(self.count_of_x_in_batch)

    def update_h(self):
        """Calculate the update step based on gradient"""
        """∇E ≈ (ŵ i(h) - vi)^T"""
        self.d_wi -= 1. / self.y_nums
        """update ∇E by Adam algo"""
        self.d_adam_m *= 0.9
        self.d_adam_m += 0.1 * self.d_wi
        self.d_adam_v *= 0.999
        self.d_adam_v += 0.001 * torch.mul(self.d_wi, self.d_wi)
        torch.mul(torch.div(self.d_adam_m, torch.add(torch.sqrt(self.d_adam_v), 1e-8)), -self.lr, out=self.d_delta_h)
        torch.add(self.d_h, self.d_delta_h, out=self.d_h)
        """∇h = ∇h - mean(∇h)"""
        self.d_h -= torch.mean(self.d_h)

    def run_gradient_descent(self, last_step=0, init_sample_batch=1):
        """Gradient descent method. Update self.d_h to the optimal solution.
        Args:
            last_step: Iteration performed before the calling. Used when resuming the training. Default [0].
            init_sample_batch: initial batch number for Monte-Carlo samples. Value of num_bat will increase during iteration. Default [1].
                     total number of MC samples used in each iteration = self.batch_size_x * num_bat
        Returns:
            self.d_h: Optimal value of h (the variable to be optimized of the variational Energy).
        """
        curr_best_wi_norm = 1e20
        steps = 0
        count_bad = 0
        dyn_sample_batch = init_sample_batch
        h_file_list = []
        m_file_list = []
        v_file_list = []

        while steps <= self.max_iter:
            self.qrng.reset()
            self.d_wi_sum.fill_(0.)
            for count in range(dyn_sample_batch):
                self.generate_samples(count)
                self.calculate_energy_for_sampled_x()
                torch.add(self.d_wi_sum, self.d_wi, out=self.d_wi_sum)

            # calculate avg
            torch.div(self.d_wi_sum, dyn_sample_batch, out=self.d_wi)
            self.update_h()

            # 预期 wi == vi，(vi是离散分布，1/N), 所以预期wi是均匀分布，所以使用?
            #
            # √Σ(wi-vi)^2
            wi_diff_error = torch.sqrt(torch.sum(torch.mul(self.d_wi, self.d_wi)))

            torch.abs(self.d_wi, out=self.d_wi)
            wi_ratio = torch.max(self.d_wi) * self.y_nums

            print('train ot [{0}/{1}] max absolute error ratio: {2:.3f}. √Σ(wi-vi)^2 : {3:.6f}'.format(
                steps, self.max_iter, wi_ratio, wi_diff_error))

            if dyn_sample_batch > 1e6:
                self.save_state(h_file_list, last_step, m_file_list, steps, v_file_list)
                return

            self.save_state(h_file_list, last_step, m_file_list, steps, v_file_list)

            if wi_diff_error <= curr_best_wi_norm:
                curr_best_wi_norm = wi_diff_error
                count_bad = 0
            else:
                count_bad += 1
            if count_bad > 30:
                dyn_sample_batch *= 2
                print('samples has increased to {}'.format(dyn_sample_batch * self.count_of_x_in_batch))
                count_bad = 0
                curr_best_wi_norm = 1e20

            steps += 1

    def save_state(self, h_file_list, last_step, m_file_list, steps, v_file_list):
        pt_filename = '{}.pt'.format(steps + last_step)
        torch.save(self.d_h, self.model_path('h', pt_filename))
        torch.save(self.d_adam_m, self.model_path('adam_m', pt_filename))
        torch.save(self.d_adam_v, self.model_path('adam_v', pt_filename))
        h_file_list.append(self.model_path('h', pt_filename))
        m_file_list.append(self.model_path('adam_m', pt_filename))
        v_file_list.append(self.model_path('adam_v', pt_filename))
        if len(h_file_list) > 5:
            if os.path.exists(h_file_list[0]):
                os.remove(h_file_list[0])
            h_file_list.pop(0)
            if os.path.exists(v_file_list[0]):
                os.remove(v_file_list[0])
            v_file_list.pop(0)
            if os.path.exists(m_file_list[0]):
                os.remove(m_file_list[0])
            m_file_list.pop(0)

    def set_h(self, h_tensor):
        self.d_h.copy_(h_tensor)

    def set_adam_m(self, adam_m_tensor):
        self.d_adam_m.copy_(adam_m_tensor)

    def set_adam_v(self, adam_v_tensor):
        self.d_adam_v.copy_(adam_v_tensor)

    def model_path(self, sub: str, file=""):
        return os.path.join(self.model_root_path, sub, file)

    def h_path(self):
        return os.path.join(self.model_root_path, 'h_final.pt')


def load_last_file(path, file_ext):
    if not os.path.exists(path):
        os.makedirs(path)
        return None, None
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    def get_index(f: str):
        return int(f.split('.')[0])
    sorted_files = sorted(files, key=get_index)
    if not sorted_files:
        return None, None
    else:
        last = sorted_files[-1]
        last_f_id, last_f = get_index(last), os.path.join(path, last)
        print('Last' + path + ': ', last_f_id)
        for f in sorted_files[:-10]:
            old = os.path.join(path, f)
            if os.path.exists(old):
                os.remove(old)
        return last_f_id, last_f


def train_omt(model: OMTRaw, init_sample_batch=1):
    """
    train uₕ(x) function, to find the optimal hi
    """
    last_step = 0
    '''load last trained model parameters and last omt parameters'''
    h_id, h_file = load_last_file(model.model_path('h'), '.pt')
    adam_m_id, m_file = load_last_file(model.model_path('adam_m'), '.pt')
    adam_v_id, v_file = load_last_file(model.model_path('adam_v'), '.pt')
    if h_id is not None:
        if h_id != adam_m_id or h_id != adam_v_id:
            sys.exit('Error: h, adam_m, adam_v file log does not match')
        elif h_id is not None and adam_m_id is not None and adam_v_id is not None:
            last_step = h_id
            model.set_h(torch.load(h_file))
            model.set_adam_m(torch.load(m_file))
            model.set_adam_v(torch.load(v_file))

    '''run gradient descent'''
    model.run_gradient_descent(last_step=last_step, init_sample_batch=init_sample_batch)

    '''record result'''
    # np.savetxt(self.model_root_path + '/h_final.csv',p_s.d_h.cpu().numpy(), delimiter=',')
