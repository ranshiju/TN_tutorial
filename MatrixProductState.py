import numpy as np
import copy
import BasicFun as bf


class OBCMPS:
    """
    重要成员函数列表：
    - initialize_tensors：随机初始化张量，或输入指定张量
    - full_tensor：收缩所有辅助指标，返回MPS代表的大张量
    - c_orthogonalization：指定正交中心，对MPS进行中心正交化
    - move_center_one_step：将MPS正交中心向左或向右移动一格（MPS必须已被正交化）
    - calculate_bipartite_entanglement: 利用中心正交形式，计算纠缠谱
    """

    def __init__(self, d, chi, length):
        """
        :param d: 物理指标维数
        :param chi: 辅助指标截断维数
        :param length: 张量个数
        注：
        1. 每个张量为三阶，第0个张量第0个指标维数为1，最后一个张量最后一个指标维数为1
        2. 每个张量的指标顺序为：左辅助、物理、右辅助
               1
               |
         0  —  A  —  2
        """
        self.d = d  # physical bond dimension
        self.chi = chi  # virtual bond dimension cut-off
        self.length = length  # number of tensors
        self.tensors = np.zeros(0)
        self.pd = list()  # physical bond dimensions
        self.vd = list()  # virtual bond dimensions
        self.center = -1  # 正交中心（当c为负数时，MPS非中心正交）
        self.initialize_tensors()

    def initialize_tensors(self, tensors=None):
        if tensors is None:
            self.tensors = np.random.randn(self.length, self.chi, self.d, self.chi)
        else:
            assert tensors.shape[0] >= self.length
            assert min(tensors.shape[1], tensors.shape[3]) >= self.chi
            self.tensors = copy.deepcopy(tensors)
            self.remove_redundant_space()
        self.pd = [self.d] * self.length
        self.vd = [1] + [self.chi] * (self.length - 1) + [1]

    def remove_redundant_space(self):
        pd = max(self.pd)
        vd = max(self.vd)
        self.tensors = self.tensors[:self.length, :vd, :pd, :vd]

    def full_tensor(self):
        tensor = self.get_tensor(0, True)
        for n in range(1, self.length):
            tensor_ = self.get_tensor(n, True)
            tensor = np.tensordot[tensor, tensor_, [[-1], [0]]]
        return np.squeeze(tensor)

    def get_tensor(self, nt, if_copy=True):
        if if_copy:
            return copy.deepcopy(self.tensors[nt, :self.vd[nt],
                                 :self.pd[nt], :self.vd[nt+1]])
        else:
            return self.tensors[nt, :self.vd[nt], :self.pd[nt], :self.vd[nt + 1]]

    def update_tensor(self, nt, tensor):
        s = tensor.shape
        self.tensors[nt, :s[0], :s[1], :s[2]] = tensor[:, :, :]

    def orthogonalize_left2right(self, nt, way, dc=-1, normalize=False):
        # dc=-1意味着不进行裁剪
        if 0 < dc < self.vd[nt + 1]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False

        tensor = self.get_tensor(nt, False)
        tensor = tensor.reshape(self.vd[nt]*self.pd[nt], self.vd[nt+1])
        if way.lower() == 'svd':
            u, lm, v = np.linalg.svd(tensor)
            if if_trun:
                u = u[:, :dc]
                r = np.diag(lm[:dc]).dot(v[:, :dc].T)
            else:
                r = np.diag(lm).dot(v.T)
        else:
            u, r = np.linalg.qr(tensor)
            lm = None
        u = u.reshape(self.vd[nt], self.pd[nt], -1)
        self.update_tensor(nt, u)
        if normalize:
            r /= np.linalg.norm(r)
        tensor_ = self.get_tensor(nt+1, False)
        tensor_ = np.tensordot(r, tensor_, [[1], [0]])
        self.update_tensor(nt+1, tensor_)
        self.vd[nt + 1] = r.shape[0]
        return lm

    def orthogonalize_right2left(self, nt, way, dc=-1, normalize=False):
        # dc=-1意味着不进行裁剪
        if 0 < dc < self.vd[nt + 1]:
            # In this case, truncation is required
            way = 'svd'
            if_trun = True
        else:
            if_trun = False
        tensor = self.get_tensor(nt, False)
        tensor = tensor.reshape(self.vd[nt], self.pd[nt]*self.vd[nt+1]).T
        if way.lower() == 'svd':
            u, lm, v = np.linalg.svd(tensor)
            if if_trun:
                u = u[:, :dc]
                r = np.diag(lm[:dc]).dot(v[:, :dc].T)
            else:
                r = np.diag(lm).dot(v.T)
        else:
            u, r = np.linalg.qr(tensor)
            lm = None
        u = u.T.reshape(-1, self.pd[nt], self.vd[nt+1])
        self.update_tensor(nt, u)
        if normalize:
            r /= np.linalg.norm(r)
        tensor_ = self.get_tensor(nt-1, False)
        tensor_ = np.tensordot(tensor_, r, [[2], [1]])
        self.update_tensor(nt-1, tensor_)
        self.vd[nt - 1] = r.shape[0]
        return lm

    def orthogonalize_n1_n2(self, n1, n2, way, dc, normalize):
        if n1 < n2:
            for nt in range(n1, n2, 1):
                self.orthogonalize_left2right(nt, way, dc, normalize)
        else:
            for nt in range(n1, n2, -1):
                self.orthogonalize_right2left(nt, way, dc, normalize)

    def c_orthogonalization(self, c, way, dc, normalize):
        if self.center < -0.5:
            self.orthogonalize_n1_n2(0, c, way, dc, normalize)
            self.orthogonalize_n1_n2(self.length - 1, c, way, dc, normalize)
        elif self.center != c:
            self.orthogonalize_n1_n2(self.center, c, way, dc, normalize)
        if normalize:
            self.normalize_central_tensor()
        self.center = c

    def move_center_one_step(self, direction, decomp_way, dc, normalize):
        if direction.lower() in ['left', 'l']:
            if self.center > 0:
                self.orthogonalize_left2right(self.center, decomp_way, dc, normalize)
                self.center += 1
            else:
                print('Error: cannot move center left as center = ' + str(self.center))
        elif direction.lower() in ['right', 'r']:
            if -0.5 < self.center < self.length-1:
                self.orthogonalize_right2left(self.center, decomp_way, dc, normalize)
                self.center -= 1
            else:
                print('Error: cannot move center right as center = ' + str(self.center))

    def normalize_central_tensor(self):
        if self.center > -0.5:
            nt = self.center
            norm = np.linalg.norm(self.tensors[nt, :self.vd[nt], :self.pd[nt], :self.vd[nt + 1]])
            self.tensors[nt, :self.vd[nt], :self.pd[nt], :self.vd[nt + 1]] /= norm

    def evolve_and_truncate_two_body_nn_gate(self, gate, nt, n_center=None):
        """
         0    1
          \  /
          gate
          /  \
         2    3
        :param gate: the two-body gate to evolve the MPS
        :param nt: the position of the first spin in the gate
        :param n_center: where to put the new center; nt (None) or nt+1
        :return:
        """
        if self.center <= nt:
            self.c_orthogonalization(nt, 'qr', -1, True)
        else:
            self.c_orthogonalization(nt+1, 'qr', -1, True)
        tensor1 = self.get_tensor(nt)
        tensor2 = self.get_tensor(nt+1)
        tensor = np.einsum('iba,acj,klbc->iklj', tensor1, tensor2, gate)
        s = tensor.shape
        u, lm, v = np.linalg.svd(tensor.reshape(s[0]*s[1], -1))
        chi = min(self.chi, lm.size)
        if n_center is None or n_center == nt:
            u = u[:, :chi].dot(np.diag(lm[:chi])).reshape(s[0], s[1], chi)
            v = v[:, :chi].T.reshape(chi, s[2], s[3])
            self.center = nt
        else:
            u = u[:, :chi].reshape(s[0], s[1], chi)
            v = v[:, :chi].dot(np.diag(lm[:chi])).reshape(s[2], s[3], chi).transpose(2, 0, 1)
            self.center = nt+1
        self.update_tensor(nt, u)
        self.update_tensor(nt+1, v)
        self.vd[nt + 1] = chi

    def calculate_bipartite_entanglement(self, nt):
        # 从第nt个张量右边断开，计算纠缠
        # 计算过程中，会对MPS进行规范变换，且会自动进行归一化
        if self.center <= nt:
            self.c_orthogonalization(nt, 'qr', dc=-1, normalize=True)
            tensor = self.get_tensor(nt, True)
            lm = np.linalg.svd(tensor.reshape(
                -1, tensor.shape[2]), compute_uv=False)
        else:
            self.c_orthogonalization(nt + 1, 'qr', dc=-1, normalize=True)
            tensor = self.get_tensor(nt + 1, True)
            lm = np.linalg.svd(tensor.reshape(
                tensor.shape[0], -1), compute_uv=False)
        return lm/np.linalg.norm(lm)

    def calculate_one_body_RDM(self, nt):
        """
        :param nt: 计算第nt个自旋对应的单体约化密度矩阵
        :return rho: 约化密度矩阵
        """
        if self.center < -0.5:
            # 这种情况下，MPS不具备中心正交形式
            vl = np.ones((1, 1))
            for n in range(nt):
                tensor = self.get_tensor(n, True)
                vl = np.einsum('apb,cpd,ac->bd', tensor.conj(), tensor, vl)
                vl /= np.linalg.norm(vl)
            vr = np.ones((1, 1))
            for n in range(self.length-1, nt, -1):
                tensor = self.get_tensor(n, True)
                vr = np.einsum('apb,cpd,bd->ac', tensor.conj(), tensor, vl)
                vr /= np.linalg.norm(vr)
            tensor = self.get_tensor(nt, True)
            rho = np.einsum('apb,cqd,ac,bd->pq', tensor.conj(), tensor, vl, vr)
        else:
            if self.center < nt:
                v = np.eye(self.vd[self.center])
                for n in range(self.center, nt):
                    tensor = self.get_tensor(n, True)
                    v = np.einsum('apb,cpd,ac->bd', tensor.conj(), tensor, v)
                    v /= np.linalg.norm(v)
                tensor = self.get_tensor(nt, True)
                rho = np.einsum('apb,cqb,ac->pq', tensor.conj(), tensor, v)
            else:
                v = np.eye(self.vd[nt+1])
                for n in range(nt, self.center, -1):
                    tensor = self.get_tensor(n, True)
                    v = np.einsum('apb,cpd,bd->ac', tensor.conj(), tensor, v)
                    v /= np.linalg.norm(v)
                tensor = self.get_tensor(self.center, True)
                rho = np.einsum('apb,aqd,bd->pq', tensor.conj(), tensor, v)
        return rho / np.trace(rho)

    def calculate_one_body_observable(self, nt, op):
        rho = self.calculate_one_body_RDM(nt)
        return np.trace(rho.dot(op))

