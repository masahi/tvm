# pylint: disable=invalid-name,unused-variable,invalid-name
import tvm
from .. import generic, tag
from .. import nn
from ..nn.util import infer_pad, infer_stride
from .. import util
from ..nn import pad
from ..nn.conv2d import conv2d_winograd_without_filter_transform, winograd_filter_transform, conv2d_replace_with_winograd
from injective import schedule_injective

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

@conv2d_replace_with_winograd.register("cpu")
def replace_with_winograd_4x4(attrs, inputs, tinfos):
    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    #new_attrs = {k : attrs[k] for k in attrs.keys()}
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    channels = attrs.get_int("channels")
    in_channel = inputs[0].shape[1]

    if groups == 1 and kernel_size == (3, 3) and strides == (1, 1) and padding == (1, 1) and channels >= 8 and in_channel >= 8:
        new_attrs = {}
        new_attrs['use_gpu'] = false;
        new_attrs['tile_size'] = 6;
        copy_inputs[1] = sym.contrib.winograd_filter_transform(*news_attrs, tvm.Array(inputs[1]), tinfos)        
        new_attrs['layout'] = 'NCHW8c'
        new_attrs['out_layout'] = 'NCHW8c'
        new_attrs['kernel_layout'] = 'OIHW'
        new_attrs['channels'] = channel
        new_attrs['use_bias'] = attrs.get_bool("use_bias")
        return sym.contrib.conv2d_winograd_without_filter_transform(*copy_inputs, **new_attrs)
    return sym.conv2d(attrs, inputs, tinfos)

@winograd_filter_transform.register("cpu")
def winograd_filter_transform_4x4(attrs, inputs, tinfos):
    shape = util.get_const_tuple(kernel.shape)
    shape = (6, 6) + shape[:2]
    G_data = np.array([
        [1 / 4.0, 0, 0],
        [-1 / 6.0, -1 / 6.0, -1 / 6.0],
        [-1 / 6.0, 1 / 6.0, -1 / 6.0],
        [1 / 24.0, 1 / 12.0, 1 / 6.0],
        [1 / 24.0, -1 / 12.0, 1 / 6.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    G = const_matrix(G_data, 'G')
    r_kh = tvm.reduce_axis((0, 3), name='r_kh')
    r_kw = tvm.reduce_axis((0, 3), name='r_kw')
    return tvm.compute(shape, lambda eps, nu, co, ci:
                       tvm.sum(kernel[co][ci][r_kh][r_kw] *
                               G[eps][r_kh] * G[nu][r_kw],
                               axis=[r_kh, r_kw]), name='transform_weight')

@conv2d_winograd_without_filter_transform.register("cpu")
def conv2d_winograd_4x4(data, U):
    N, co, H, W, ci = [util.get_const_int(x) for x in data.shape]
    co, ko, _, _, ci, ki  = [util.get_const_int(x) for x in U.shape]
    C = co * ci
    K = ko * ki
    HPAD, WPAD = 1,1
    HSTR, WSTR = 1,1
    out_dtype = "float32"

    data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")

    B_data = np.array([
        [4, 0, 0, 0, 0, 0],
        [0, -4, 4, -2, 2, 4],
        [-5, -4, -4, -1, -1, 0],
        [0, 1, -1, 2, -2, -5],
        [1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], out_dtype)

    A_data = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 2, 4, 8],
        [1, -2, 4, -8],
        [0, 0, 0, 1]
    ], out_dtype)
    
    m = 4
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8

    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((P // bnb, C // bna, alpha, alpha, bnb, bna), lambda b, c, eps, nu, bb, cc:
                    tvm.sum(data_pad[(b*bnb+bb) // (nH*nW)][c][(b*bnb+bb) // nW % nH * m + r_eps][(b*bnb+bb) % nW * m + r_nu][cc] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')
    
    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((P //bnb, K // bna, alpha, alpha, bnb, bna), lambda b, k, eps, nu, bb, kk:
                    tvm.sum(V[b][c // bna][eps][nu][bb][c % bna] *
                            U[c // bna][k][eps][nu][c % bna][kk], axis=c), name='M')
    
    # inverse transform
    A = const_array(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    output = tvm.compute((N, K // bna, H, W, bna), lambda n, k, h, w, kk: 
                    tvm.sum(M[(n * nH * nW + (h//m) * nW + w//m)//bna][k][r_eps][r_nu][(n * nH * nW + (h//m) * nW + w//m)%bna][kk] * A[r_eps][h % m] * A[r_nu][w % m],
                            axis=[r_eps, r_nu]), name='output', tag=tag="conv2d_winograd")
    
    return output

@generic.schedule_winograd_filter_transform.register(["cpu"]):
def schedule_winograd_weight_transform(outs):
    return schedule_injective(outs)

def schedule_winograd_without_filter_transform(s, op):
    output = op.output(0)
    M, A = s[output].op.input_tensors
    V, U = s[M].op.input_tensors
    data_pad, B = s[V].op.input_tensors
    data = s[data_pad].op.input_tensors[0]

    # transform image
    s[data_pad].compute_inline()    
    s[B].compute_inline()
    b, c, eps, nu, bb, cc = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, bb, cc, eps, nu, r_nu, r_eps)
    s[V].vectorize(cc)
    _ = [s[V].unroll(x) for x in [eps, nu, r_eps, r_nu]]
    fused = s[V].fuse(b, c)
    s[V].parallel(fused)

    # batch gemm
    b, k, eps, nu, bb, kk = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    fused = s[M].fuse(b, k)
    s[M].parallel(fused)
    co, ci = s[M].split(c, factor=8)    
    s[M].reorder(co, bb, ci, kk)
    s[M].unroll(ci)
    s[M].vectorize(kk)
    
#     # inverse transform
    s[A].compute_inline()
    n, k, h, w, kk = s[output].op.axis
    r_eps, r_nu = s[output].op.reduce_axis    
    ho, hi = s[output].split(h, factor=8)
    wo, wi = s[output].split(w, factor=8)
    s[output].reorder(n, k, ho, wo, hi, r_eps, r_nu, wi, kk)
    s[output].vectorize(kk)
    fused = s[output].fuse(n, k, ho, wo) 
    s[output].parallel(fused)
    _ = [s[output].unroll(x) for x in [r_eps, r_nu]]

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
            schedule_winograd_without_filter_transform(s, op)

    traverse(output_op)
    return s
    
