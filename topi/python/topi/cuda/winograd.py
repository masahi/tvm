# pylint: disable=invalid-name,unused-variable,invalid-name
import tvm
import numpy as np
from .. import generic, tag
from .. import nn
from ..nn.util import infer_pad, infer_stride
from .. import util
from ..nn import pad
from ..nn.conv2d import conv2d_winograd_without_filter_transform, winograd_filter_transform, conv2d_replace_with_winograd
from .injective import schedule_injective

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

@conv2d_replace_with_winograd.register(["cuda", "gpu"])
def replace_with_winograd_2x2(attrs, inputs, tinfos):
    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    #new_attrs = {k : attrs[k] for k in attrs.keys()}
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    channels = attrs.get_int("channels")
    kernel_size = attrs.get_int_tuple("kernel_size")    
    in_channel = tinfos[0].shape[1].value

    if groups == 1 and kernel_size == (3, 3) and strides == (1, 1) and padding == (1, 1) and channels >= 8 and in_channel >= 8:
        new_attrs = {}
        copy_inputs[1] = sym.contrib.winograd_filter_transform(inputs[1], tile_size=4, use_gpu=True)
        new_attrs['use_gpu'] = True
        new_attrs['tile_size'] = 4
        new_attrs['layout'] = 'NCHW'
        new_attrs['out_layout'] = 'NCHW'
        new_attrs['kernel_layout'] = 'OIHW'
        new_attrs['channels'] = channels
        new_attrs['use_bias'] = attrs.get_bool("use_bias")
        return sym.contrib.conv2d_winograd_without_filter_transform(*copy_inputs, **new_attrs)

    return sym.conv2d(attrs, inputs, tinfos)

@winograd_filter_transform.register("cpu")
def winograd_filter_transform_2x2(kernel):
    shape = util.get_const_tuple(kernel.shape)
    shape = (4, 4) + shape[:2]
    G_data = np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1],
    ], np.float32)
    
    G = const_array(G_data, 'G')
    r_kh = tvm.reduce_axis((0, 3), name='r_kh')
    r_kw = tvm.reduce_axis((0, 3), name='r_kw')
    C = kernel.shape[1].value
    K = kernel.shape[0].value
    bna, bnb = 4, 4 
    U = tvm.compute((4, 4, K // bna, C, bna), lambda eps, nu, k, c, kk:
                    tvm.sum(kernel[k * bna + kk][c][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                            axis=[r_kh, r_kw]), name='U')
    return U

@conv2d_winograd_without_filter_transform.register(["cuda", "gpu"])
def conv2d_winograd_2x2(data, U):
    N, co, H, W, ci = [util.get_const_int(x) for x in data.shape]
    co, ko, _, _, ci, ki  = [util.get_const_int(x) for x in U.shape]
    C = co * ci
    K = ko * ki
    HPAD, WPAD = 1,1
    HSTR, WSTR = 1,1
    out_dtype = "float32"

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
                            axis=[r_eps, r_nu]), name='output', tag='conv2d_winograd')

    return output

@generic.schedule_winograd_filter_transform.register(["gpu", "cuda"])
def schedule_winograd_weight_transform(outs):
    return schedule_injective(outs)

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

def schedule_winograd_without_filter_transform(s, conv_op, output_op):
    output = conv_op.output(0)
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
    
    if conv_op == output_op:
        pass
        # fused = s[output].fuse(n, k, ho, wo)
        # s[output].parallel(fused)
    else:
        pass
        # n, k, h, w, kk = s[output_op].op.axis
        # ho, hi = s[output_op].split(h, factor=8)
        # wo, wi = s[output_op].split(w, factor=8)
        # s[output_op].reorder(n, k, ho, wo, hi, wi, kk)
        # s[output_op].vectorize(kk)
        # fused = s[output_op].fuse(n, k, ho, wo) 
        # s[output_op].parallel(fused)
        # s[conv_op].compute_at(s[output_op], fused)
        
    return s

@generic.schedule_conv2d_winograd_without_filter_transform.register(["cpu"])
def schedule_winograd(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op


    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'conv2d_winograd' in op.tag:
            schedule_winograd_without_filter_transform(s, op, output_op)

    traverse(output_op)
    return s
    
