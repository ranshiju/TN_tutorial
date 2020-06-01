import numpy as np
import torch as tc
import copy
from scipy.sparse.linalg import eigsh


def eig0(mat, it_time=100, tol=1e-15):
    """
    :param mat: 输入矩阵（实对称阵）
    :param it_time: 最大迭代步数
    :param tol: 收敛阈值
    :return lm: （绝对值）最大本征值
    :return v1: 最大本征向量
    """
    # 初始化向量
    v1 = np.random.randn(mat.shape[0],)
    v0 = copy.deepcopy(v1)
    lm = 1
    for n in range(it_time):  # 开始循环迭代
        v1 = mat.dot(v0)  # 计算v1 = M V0
        lm = np.linalg.norm(v1)  # 求本征值
        v1 /= lm  # 归一化v1
        # 判断收敛
        conv = np.linalg.norm(v1 - v0)
        if conv < tol:
            break
        else:
            v0 = copy.deepcopy(v1)
    return lm, v1


def svd0(mat, it_time=100, tol=1e-15):
    """
    Recursive algorithm to find the dominant singular value and vectors
    :param mat: input matrix (assume to be real)
    :param it_time: max iteration time
    :param tol: tolerance of error
    :return u: the dominant left singular vector
    :return s: the dominant singular value
    :return v: the dominant right singular vector
    """
    dim0, dim1 = mat.shape
    # 随机初始化奇异向量
    u, v = np.random.randn(dim0, ), np.random.randn(dim1, )
    # 归一化初始向量
    u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
    s = 1

    for t in range(it_time):
        # 更新v和s
        v1 = u.dot(mat)
        s1 = np.linalg.norm(v1)
        v1 /= s1
        # 更新u和s
        u1 = mat.dot(v1)
        s1 = np.linalg.norm(u1)
        u1 /= s1
        # 计算收敛程度
        conv = np.linalg.norm(u - u1) / dim0 + np.linalg.norm(v - v1) / dim1
        u, s, v = u1, s1, v1
        # 判断是否跳出循环
        if conv < tol:
            break
    return u, s, v


def rank1decomp(x, it_time=100, tol=1e-15):
    """
    :param x: 待分解的张量
    :param it_time: 最大迭代步数
    :param tol: 迭代终止的阈值
    :return vs: 储存rank-1分解各个向量的list
    :return k: rank-1系数
    """
    ndim = x.ndim  # 读取张量x的阶数
    dims = x.shape  # 读取张量x各个指标的维数

    # 初始化vs中的各个向量并归一化
    vs = list()  # vs用以储存rank-1分解得到的各个向量
    for n in range(ndim):
        _v = np.random.randn(dims[n])
        vs.append(_v / np.linalg.norm(_v))
    k = 1

    for t in range(it_time):
        vs0 = copy.deepcopy(vs)  # 暂存各个向量以计算收敛情况
        for _ in range(ndim):
            # 收缩前(ndim-1)个向量，更新最后一个向量
            x1 = copy.deepcopy(x)
            for n in range(ndim-1):
                x1 = np.tensordot(x1, vs[n], [[0], [0]])
            # 归一化得到的向量，并更新常数k
            k = np.linalg.norm(x1)
            x1 /= k
            # 将最后一个向量放置到第0位置
            vs.pop()
            vs.insert(0, x1)
            # 将张量最后一个指标放置到第0位置
            x = x.transpose([ndim-1] + list(range(ndim-1)))
        # 计算收敛情况
        conv = np.linalg.norm(np.hstack(vs0) - np.hstack(vs))
        if conv < tol:
            break
    return vs, k


def hosvd(x):
    """
    :param x: 待分解的张量
    :return G: 核张量
    :return U: 变换矩阵
    :return lm: 各个键约化矩阵的本征谱
    """
    ndim = x.ndim
    U = list()  # 用于存取各个变换矩阵
    lm = list()  # 用于存取各个键约化矩阵的本征谱
    x = tc.from_numpy(x)
    for n in range(ndim):
        index = list(range(ndim))
        index.pop(n)
        _mat = tc.tensordot(x, x, [index, index])
        _lm, _U = tc.symeig(_mat, eigenvectors=True)
        lm.append(_lm.numpy())
        U.append(_U)
    # 计算核张量
    G = tucker_product(x, U)
    U1 = [u.numpy() for u in U]
    return G, U1, lm


def tucker_product(x, U, dim=1):
    """
    :param x: 张量
    :param U: 变换矩阵
    :param dim: 收缩各个矩阵的第几个指标
    :return G: 返回Tucker乘积的结果
    """
    ndim = x.ndim
    if type(x) is not tc.Tensor:
        x = tc.from_numpy(x)

    U1 = list()
    for n in range(len(U)):
        if type(U[n]) is not tc.Tensor:
            U1.append(tc.from_numpy(U[n]))
        else:
            U1.append(U[n])

    ind_x = ''
    for n in range(ndim):
        ind_x += chr(97 + n)
    ind_x1 = ''
    for n in range(ndim):
        ind_x1 += chr(97 + ndim + n)
    contract_eq = copy.deepcopy(ind_x)
    for n in range(ndim):
        if dim == 0:
            contract_eq += ',' + ind_x[n] + ind_x1[n]
        else:
            contract_eq += ',' + ind_x1[n] + ind_x[n]
    contract_eq += '->' + ind_x1
    # print(x.shape, U[0].shape, U[1].shape, U[2].shape)
    # print(type(contract_eq), contract_eq)
    G = tc.einsum(contract_eq, [x] + U1)
    G = G.numpy()
    return G


def spin_operator_one_half():
    op = dict()
    op['i'] = np.eye(2)  # Identity
    op['x'] = np.zeros((2, 2))
    op['x'][0, 1] = 1 / 2
    op['x'][1, 0] = 1 / 2
    op['y'] = np.zeros((2, 2), dtype=np.complex)
    op['y'][0, 1] = 1j / 2
    op['y'][1, 0] = -1j / 2
    op['z'] = np.zeros((2, 2))
    op['z'][0, 0] = 1 / 2
    op['z'][1, 1] = -1 / 2
    return op


def heisenberg_hamilt(j, h):
    """
    :param j: list，耦合参数[Jx, Jy, Jz]
    :param h: list，外磁场[hx, hy, hz]
    :return H: 哈密顿量
    """
    op = spin_operator_one_half()
    H = j[0]*np.kron(op['x'], op['x']) + j[1]*np.kron(op['y'], op['y']) + \
        j[2]*np.kron(op['z'], op['z'])
    H += h[0] * (np.kron(op['x'], op['i']) + np.kron(op['i'], op['x']))
    H += h[1] * (np.kron(op['y'], op['i']) + np.kron(op['i'], op['y']))
    H += h[2] * (np.kron(op['z'], op['i']) + np.kron(op['i'], op['z']))
    if np.linalg.norm(np.imag(H)) < 1e-20:
        H = np.real(H)
    return H


def ED_ground_state(hamilt, pos, v0=None, tau=1e-4):
    """
    每个局域哈密顿量的指标顺序满足: (bra0, bra1, ..., ket0, ket1, ...)
    例：求单个三角形上定义的反铁磁海森堡模型基态：
    H2 = heisenberg_hamilt([1, 1, 1], [0, 0, 0])
    e0, gs = ED_ground_state([H2.reshape(2, 2, 2, 2)]*3, [[0, 1], [1, 2], [0, 2]])
    print(e0)

    :param hamilt: list，局域哈密顿量
    :param pos: 每个局域哈密顿量作用的自旋
    :param v0: 初态
    :param tau: 平移量 H <- I - tau*H
    :return lm: 最大本征值
    :return v1: 最大本征向量
    """
    from scipy.sparse.linalg import LinearOperator as LinearOp

    def convert_nums_to_abc(nums, n0=0):
        s = ''
        n0 += 97
        for m in nums:
            s += chr(m + n0)
        return s

    def one_map(v, hs, pos_hs, tau_shift, v_dims, ind_v, ind_v_str):
        v = v.reshape(v_dims)
        _v = copy.deepcopy(v)
        for n, p in enumerate(pos_hs):
            ind_h = list()
            for nn in range(len(p)):
                ind_h.append(ind_v.index(p[nn]))
            ind_h1 = convert_nums_to_abc(ind_h)
            ind_h2 = convert_nums_to_abc(list(range(len(p))), n0=len(p))
            ind_f_str = list(copy.deepcopy(ind_v_str))
            for nn, _ind in enumerate(ind_h):
                ind_f_str[_ind] = ind_h2[nn]
            ind_f_str = ''.join(ind_f_str)
            eq = ind_v_str + ',' + ind_h1 + ind_h2 + '->' + ind_f_str
            _v = _v - tau_shift * np.einsum(eq, v, hs[n])
        return _v.reshape(-1, )

    # 自动获取总格点数
    n_site = 0
    for x in pos:
        n_site = max([n_site] + list(x))
    n_site += 1
    # 自动获取格点自由度
    d = hamilt[0].shape[0]
    dims = [d] * n_site
    dim_tot = np.prod(dims)
    # 初始化向量
    if v0 is None:
        v0 = eval('np.random.randn' + str(tuple(dims)))
        # v0 = np.random.randn(dims)
    else:
        v0 = v0.reshape(dims)
    v0 /= np.linalg.norm(v0)
    # 初始化指标顺序
    ind = list(range(n_site))
    ind_str = convert_nums_to_abc(ind)
    # 定义等效线性映射：I - tau*H
    h_effect = LinearOp((dim_tot, dim_tot), lambda vg: one_map(
        vg, hamilt, pos, tau, dims, ind, ind_str))
    lm, v1 = eigsh(h_effect, k=1, which='LM', v0=v0)
    # 平移本征值
    lm = (1 - lm) / tau
    return lm, v1


def tt_product(tensors):
    """
    Tensor-train product
    :param tensors: tensors in the TT form
    :return: tensor
    """
    x = np.tensordot(tensors[0], tensors[1], [[tensors[0].ndim-1], [0]])
    for n in range(len(tensors)-2):
        x = np.tensordot(x, tensors[n+2], [[x.ndim - 1], [0]])
    return x


def ttd(x, chi=None):
    """
    :param x: tensor to be decomposed
    :param chi: dimension cut-off. Use QR decomposition when chi=None;
                use SVD but don't truncate when chi=-1
    :return tensors: tensors in the TT form
    :return lm: singular values in each decomposition (calculated when chi is not None)
    """
    dims = x.shape
    ndim = x.ndim
    dimL = 1
    tensors = list()
    lm = list()
    for n in range(ndim-1):
        if chi is None:  # No truncation
            q, x = np.linalg.qr(x.reshape(dimL*dims[n], -1))
            dimL1 = x.shape[0]
        else:
            q, s, v = np.linalg.svd(x.reshape(dimL*dims[n], -1))
            if chi > 0:
                dc = min(chi, s.size)
            else:
                dc = s.size
            q = q[:, :dc]
            s = s[:dc]
            lm.append(s)
            x = np.diag(s).dot(v[:dc, :])
            dimL1 = dc
        tensors.append(q.reshape(dimL, dims[n], dimL1))
        dimL = dimL1
    tensors.append(x.reshape(dimL, dims[-1]))
    tensors[0] = tensors[0][0, :, :]
    return tensors, lm


# x = np.random.randn(10, 10, 10)
# ts = ttd(x, chi=3)[0]
# x1 = tt_product(ts)
# print(np.linalg.norm(x-x1) / np.linalg.norm(x))



