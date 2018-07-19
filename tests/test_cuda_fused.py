import numpy as np
import tvm
from tvm.contrib import graph_runtime as runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
import os
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi import util
from topi.nn import pad

def const_array(data, name):
    """ convert an const array to tvm tensor"""
    row, col = data.shape
    dtype = str(data.dtype)

    def select_array(i, j):
        now = tvm.const(0.0, dtype)
        for ii in range(row):
            for jj in range(col):
                now = tvm.select(tvm.all(i % row == ii, j % col == jj),
                                 tvm.const(data[ii][jj], dtype),
                                 now)
        return now
    return tvm.compute(data.shape, select_array, name=name)

def decl_winograd(data, U, stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, C, H, W = [util.get_const_int(x) for x in data.shape]
    _, _, C, K = [util.get_const_int(x) for x in U.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    B_data = np.array([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]
    ], out_dtype)

    A_data = np.array([
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1],
    ], out_dtype)

    m = 2
    r = 3
    alpha = m + r - 1
    K = K

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    # pack input tile
    input_tile = tvm.compute((C, P, alpha, alpha),
                             lambda c, b, eps, nu:
                             tvm.select(b < P, data_pad[b // (nH*nW)][c][b// nW % nH * m + eps][b % nW * m + nu], tvm.const(0, data_pad.dtype)), name='d')

    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha, alpha, C, P), lambda eps, nu, c, b:
                    tvm.sum(input_tile[c][b][r_eps][r_nu] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P), lambda eps, nu, k, b:
                    tvm.sum(U[eps][nu][c][k] *
                            V[eps][nu][c][b], axis=c), name='M')

    # inverse transform and unpack
    A = const_array(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                    tvm.sum(M[r_eps][r_nu][k][n * nH * nW + (h//m) * nW + w//m] * A[r_eps][h % m] * A[r_nu][w % m],
                            axis=[r_eps, r_nu]), name='output')

    return output

def schedule_smem_load(s, smem, num_thread):
    yi, xi, ci, ni = s[smem].op.axis
    ty, ci = s[smem].split(ci, nparts=num_thread)
    tx, ni = s[smem].split(ni, nparts=num_thread)
    _, ni = s[smem].split(ni, factor=4)
    s[smem].reorder(ty, tx, yi, xi, ci, ni)
    s[smem].vectorize(ni)  # vectorize memory load
    s[smem].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[smem].bind(tx, tvm.thread_axis("threadIdx.x"))

def schedule_batched_sgemm(s, U, V, M):
    UU = s.cache_read(U, 'shared', [M])
    VV = s.cache_read(V, "shared", [M])
    UL = s.cache_read(UU, "local", [M])
    VL = s.cache_read(VV, "local", [M])
    ML = s.cache_write(M, "local")

    tile = 8
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

    eps, nu, k, p = s[M].op.axis
    ko, ki = s[M].split(k, factor=block_factor)
    po, pi = s[M].split(p, factor=block_factor)
    z = s[M].fuse(eps, nu)

    s[M].bind(z, tvm.thread_axis("blockIdx.z"))
    s[M].bind(ko, tvm.thread_axis("blockIdx.y"))
    s[M].bind(po, tvm.thread_axis("blockIdx.x"))

    tyz, kii = s[M].split(ki, nparts=vthread)  # virtual thread split
    txz, pii = s[M].split(pi, nparts=vthread)  # virtual thread split
    ty, kii = s[M].split(kii, nparts=num_thread)
    tx, pii = s[M].split(pii, nparts=num_thread)
    s[M].reorder(z, ko, po, tyz, txz, ty, tx, kii, pii)

    s[M].bind(tyz, thread_yz)
    s[M].bind(txz, thread_xz)
    s[M].bind(ty, thread_y)
    s[M].bind(tx, thread_x)

    s[ML].compute_at(s[M], tx)
    eps, nu, k, p = s[ML].op.axis
    c = s[ML].op.reduce_axis[0]
    co, ci = s[ML].split(c, factor=step)
    s[ML].reorder(co, ci, k, p)

    s[UU].compute_at(s[ML], co)
    s[VV].compute_at(s[ML], co)
    s[UL].compute_at(s[ML], ci)
    s[VL].compute_at(s[ML], ci)

    schedule_smem_load(s, UU, num_thread)
    schedule_smem_load(s, VV, num_thread)

def schedule_winograd(outs):
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)

    M, A = s[output].op.input_tensors
    U, V = s[M].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    s[data_pad].compute_inline()

    # transform image
    s[B].compute_inline()
    VL = s.cache_write(V, "local")
    eps, nu, c, p = s[V].op.axis
    r_eps, r_nu = s[VL].op.reduce_axis
    s[V].reorder(c, p, eps, nu)

    co, ci = s[V].split(c, factor=16)
    po, pi = s[V].split(p, factor=16)
    s[V].bind(ci, tvm.thread_axis("threadIdx.y"))
    s[V].bind(pi, tvm.thread_axis("threadIdx.x"))
    s[V].bind(co, tvm.thread_axis("blockIdx.y"))
    s[V].bind(po, tvm.thread_axis("blockIdx.x"))

    s[VL].compute_at(s[V], pi)
    s[d].compute_at(s[V], pi)

    schedule_batched_sgemm(s, U, V, M)

    # inverse transform
    s[A].compute_inline()
    n, k, h, w = s[output].op.axis
    ML = s.cache_read(M, "local", [output])
    output_L = s.cache_write(output, "local")
    ho, hi = s[output].split(h, factor=2)
    wo, wi = s[output].split(w, factor=2)
    s[output].reorder(k, n, ho, wo, hi, wi)
    k = s[output].fuse(k, n)

    hoo, hoi = s[output].split(ho, factor=16)
    woo, woi = s[output].split(wo, factor=16)
    s[output].bind(hoi, tvm.thread_axis("threadIdx.y"))
    s[output].bind(woi, tvm.thread_axis("threadIdx.x"))
    s[output].bind(hoo, tvm.thread_axis("blockIdx.y"))
    s[output].bind(woo, tvm.thread_axis("blockIdx.x"))
    s[output].bind(k, tvm.thread_axis("blockIdx.z"))
    s[output_L].compute_at(s[output], woi)
    s[ML].compute_at(s[output], woi)

    return s


def transform_filter(w_np):
    num_filter, in_channel, kernel, kernel = w_np.shape
    G = np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1],
    ], w_np.dtype)

    out = np.empty((4, 4, in_channel, num_filter), w_np.dtype)
    for i in range(in_channel):
        for j in range(num_filter):
            out[:, :, i, j] = np.dot(G, np.dot(w_np[j, i], G.transpose()))
    return out


def test_winograd(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    U = tvm.placeholder((4, 4, in_channel, num_filter), name='W')

    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype
    dilation = 1

    @memoize("topi.tests.test_topi_conv2d_nchw.wino")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    with tvm.target.create(device):
        B = decl_winograd(A, U, stride, padding, dtype)
        s = schedule_winograd([B])

    u_np = transform_filter(w_np)

    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    u = tvm.nd.array(u_np, ctx)
    b = tvm.nd.array(np.zeros(util.get_const_tuple(B.shape), dtype=B.dtype), ctx)
    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=(device != "cuda"),
                          partition_const_loop=False):
        func = tvm.build(s, [A, U, B], device)
        func(a, u, b)
        #print(tvm.lower(s, [A, U, B], simple_mode=True))
        num_runs = 100
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        return timer(a, u, b).mean



def get_sym(layout, kernel_layout, channels):
    data = sym.Variable(name="data")
    data = sym.conv2d(data=data, kernel_size=(3,3), channels=channels, padding=(1, 1),
                      layout=layout, kernel_layout=kernel_layout, use_bias=True)
    data = sym.batch_norm(data)
    data = sym.relu(data)
    return data


def build_and_run(sym, data, out_shape, opt_level=4):
    ctx = tvm.gpu(0)
    import logging
    #logging.basicConfig(level=logging.DEBUG)
    from nnvm.testing.utils import create_workload
    sym, params = create_workload(sym, 1, data.shape[1:])
    with nnvm.compiler.build_config(opt_level=opt_level):
        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=True):
            graph, lib, params = nnvm.compiler.build(sym, "nvptx", shape={"data":data.shape}, params=params)
    #print(graph.json())
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out =  module.get_output(0, tvm.nd.empty(out_shape))
    num_iter = 100
    ftimer = module.module.time_evaluator("run", ctx, num_iter)    
    return ftimer().mean


vgg_workloads = [(1, 64, 224, 64), #relu, input and output transform slow
                 (1, 64, 112, 128),#relu2
                 (1, 128, 112, 128),
                 (1, 128, 56, 256),
                 (1, 256, 56, 256), #relu4
                 (1, 256, 28, 512),
                 (1, 512, 28, 512), # relu6
                 (1, 512, 14, 512) # relu7
                 ]

fused_t = []
t = []
for workload in vgg_workloads:
    in_channel = workload[1]
    out_channel = workload[3]
    in_size = workload[2]
    data_shape = (1, in_channel, in_size, in_size) 
    nchw_sym = get_sym("NCHW", "OIHW", out_channel)
    data = np.random.uniform(-1, 1, data_shape).astype(np.float32)
    oshape = (1, out_channel, in_size, in_size)
    fused_t.append(1000 * build_and_run(nchw_sym, data, oshape, opt_level=4))
    t.append(1000 * test_winograd(*workload, 3, 1, 1, "nvptx"))

for (workload, t1, t2) in zip(vgg_workloads, t, fused_t):
    print(workload, t1, t2)
