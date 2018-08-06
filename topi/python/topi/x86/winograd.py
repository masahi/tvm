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
from .conv2d import _alter_conv2d_layout

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
    kernel_size = attrs.get_int_tuple("kernel_size")    
    in_channel = tinfos[0].shape[1].value
    in_size = tinfos[0].shape[2].value

    if groups == 1 and kernel_size == (3, 3) and strides == (1, 1) and padding == (1, 1) and channels >= 8 and in_channel >= 8 and in_size >= 16:
        print("replacing conv2d with winograd 4x4")
        new_attrs = {}
        copy_inputs[1] = sym.contrib.winograd_filter_transform(inputs[1], tile_size=6, use_gpu=False)
        new_attrs['use_gpu'] = False
        new_attrs['tile_size'] = 6
        new_attrs['layout'] = 'NCHW8c'
        new_attrs['out_layout'] = 'NCHW8c'
        new_attrs['kernel_layout'] = 'OIHW'
        new_attrs['channels'] = channels
        new_attrs['use_bias'] = attrs.get_bool("use_bias")
        return sym.contrib.conv2d_winograd_without_filter_transform(*copy_inputs, **new_attrs)

    return _alter_conv2d_layout(attrs, inputs, tinfos)

@winograd_filter_transform.register("cpu")
def winograd_filter_transform_4x4(kernel):
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
    
    G = const_array(G_data, 'G')
    r_kh = tvm.reduce_axis((0, 3), name='r_kh')
    r_kw = tvm.reduce_axis((0, 3), name='r_kw')
    C = kernel.shape[1].value
    K = kernel.shape[0].value
    bnb, bna = 8, 8
    U = tvm.compute((C // bnb, K // bna, 6,6, 8, 8), lambda c, k, eps, nu, cc, kk:
                    tvm.sum(kernel[k * bna + kk][c * bnb + cc][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                            axis=[r_kh, r_kw]), name='U')
    return U

def decl_V_minimal(data_pad, P, C, alpha, bna, bnb, nH, nW, m):
    def compute_temp(b, c, eps, nu, cc):
        temp_expr = {}
        batch_index = b // (nH*nW)
        h = b // nW % nH * m
        w = b % nW * m
        for j in range(6):
            t0 = data_pad[batch_index][c][h+4][w+j][cc] - data_pad[batch_index][c][h+2][w+j][cc]*4.0
            t1 = data_pad[batch_index][c][h+3][w+j][cc] - data_pad[batch_index][c][h+1][w+j][cc]*4.0
            t2 = data_pad[batch_index][c][h+4][w+j][cc] - data_pad[batch_index][c][h+2][w+j][cc]
            t3 = data_pad[batch_index][c][h+3][w+j][cc] - data_pad[batch_index][c][h+1][w+j][cc]
            temp_expr[(0, j)] = data_pad[batch_index][c][h+0][w+j][cc] * 4.0 - data_pad[batch_index][c][h+2][w+j][cc] * 5.0 + data_pad[batch_index][c][h+4][w+j][cc]
            temp_expr[(1, j)] = t0 + t1
            temp_expr[(2, j)] = t0 - t1
            temp_expr[(3, j)] = t2 + t3*2.0
            temp_expr[(4, j)] = t2 - t3*2.0
            temp_expr[(5, j)] = data_pad[batch_index][c][h+1][w+j][cc] * 4.0 - data_pad[batch_index][c][h+3][w+j][cc] * 5.0 + data_pad[batch_index][c][h+5][w+j][cc]

        now = tvm.const(0.0, "float32")
        for ii in range(alpha):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((P, C // bna, alpha, alpha, bna), compute_temp, name="temp_V")

    def compute_V(b, c, eps, nu, cc):
        v_expr = {}
        for i in range(6):
            t0 = temp[b][c][i][4][cc] - temp[b][c][i][2][cc]*4.0
            t1 = temp[b][c][i][3][cc] - temp[b][c][i][1][cc]*4.0
            t2 = temp[b][c][i][4][cc] - temp[b][c][i][2][cc]
            t3 = temp[b][c][i][3][cc] - temp[b][c][i][1][cc]
            v_expr[(i, 0)] = temp[b][c][i][0][cc] * 4.0 - temp[b][c][i][2][cc] * 5.0 + temp[b][c][i][4][cc]
            v_expr[(i, 1)] = t0 + t1
            v_expr[(i, 2)] = t0 - t1
            v_expr[(i, 3)] = t2 + t3*2.0
            v_expr[(i, 4)] = t2 - t3*2.0
            v_expr[(i, 5)] = temp[b][c][i][1][cc] * 4.0 - temp[b][c][i][3][cc] * 5.0 + temp[b][c][i][5][cc]

        now = tvm.const(0.0, "float32")
        for ii in range(6):
            for jj in range(6):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 v_expr[(ii, jj)],
                                 now)
        return now

    V = tvm.compute((P, C // bna, alpha, alpha, bna), compute_V, name="V")
    return V

def decl_output_minimal(M, N, K, H, W, P, alpha, bna, bnb, nH, nW, m):

    def compute_temp(b, k, eps, nu, kk):
        temp_expr = {}
        for j in range(6):
            t0 =  M[b][k][1][j][kk] + M[b][k][2][j][kk]
            t1 =  M[b][k][3][j][kk] + M[b][k][4][j][kk]
            t2 =  M[b][k][1][j][kk] - M[b][k][2][j][kk]
            t3 =  M[b][k][3][j][kk] - M[b][k][4][j][kk]
            temp_expr[(0, j)] = t0 + t1 + M[b][k][0][j][kk]
            temp_expr[(1, j)] = t2 + t3*2.0
            temp_expr[(2, j)] = t0 + t1*4.0
            temp_expr[(3, j)] = t2 + t3*8.0 + M[b][k][5][j][kk]

        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(6):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((P, K // bna, m, alpha, bna), compute_temp, name="temp_Y")

    def compute_output(b, k, eps, nu, kk):
        output_expr = {}
        for i in range(4):
            t0 =  temp[b][k][i][1][kk] + temp[b][k][i][2][kk]
            t1 =  temp[b][k][i][3][kk] + temp[b][k][i][4][kk]
            t2 =  temp[b][k][i][1][kk] - temp[b][k][i][2][kk]
            t3 =  temp[b][k][i][3][kk] - temp[b][k][i][4][kk]
            output_expr[(i, 0)] = t0 + t1 + temp[b][k][i][0][kk]
            output_expr[(i, 1)] = t2 + t3 * 2.0
            output_expr[(i, 2)] = t0 + t1 * 4.0
            output_expr[(i, 3)] = t2 + t3 * 8.0 + temp[b][k][i][5][kk]

        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(4):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 output_expr[(ii, jj)],
                                 now)
        return now

    Y = tvm.compute((P, K // bna, m, m, bna), compute_output, name="Y")
    output = tvm.compute((N, K // bna, H, W, bna), lambda n, k, h, w, kk:
                         Y[n * nH * nW + (h//m) * nW + w//m][k][h % m][w % m][kk],
                         name='output', tag='conv2d_winograd')
    
    return output

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

    m = 4
    r = 3
    alpha = m + r - 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW
    bna, bnb = 8, 8
    P_round = (P + bnb - 1) // bnb * bnb
    assert C % bna == 0, P_round % bnb == 0

    # transform image
    V = decl_V_minimal(data_pad, P, C, alpha, bna, bnb, nH, nW, m)

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((P, K // bna, alpha, alpha, bna), lambda b, k, eps, nu, kk:
                    tvm.sum(V[b][c // bna][eps][nu][c % bna] *
                            U[c // bna][k][eps][nu][c % bna][kk], axis=c), name='M')

    # inverse transform
    output = decl_output_minimal(M, N, K, H, W, P, alpha, bna, bnb, nH, nW, m)
    
    return output

@generic.schedule_winograd_filter_transform.register(["cpu"])
def schedule_winograd_weight_transform(outs):
    return schedule_injective(outs)

def schedule_winograd_without_filter_transform(s, conv_op, output_op):
    output = conv_op.output(0)
    Y = s[output].op.input_tensors[0]    
    temp_output_transform = s[Y].op.input_tensors[0]
    M = s[temp_output_transform].op.input_tensors[0]
    V, U = s[M].op.input_tensors
    temp_input_transform = s[V].op.input_tensors[0]
    data_pad = s[temp_input_transform].op.input_tensors[0]

    b_factor = 8
    P = V.shape[0].value
    if P == 16:
        b_factor = 2
    
    # transform image
    s[data_pad].compute_inline()    
    b, c, eps, nu, cc = s[V].op.axis
    bo, bi = s[V].split(b, factor=b_factor)
    s[V].reorder(bo, c, bi, eps, nu, cc)    
    s[V].vectorize(cc)
    _ = [s[V].unroll(x) for x in [eps, nu]]

    b, c, eps, nu, cc = s[temp_input_transform].op.axis
    s[temp_input_transform].vectorize(cc)
    _ = [s[temp_input_transform].unroll(x) for x in [eps, nu]]
    s[temp_input_transform].compute_at(s[V], bi)

    # batch gemm
    b, k, eps, nu, kk = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    co, ci = s[M].split(c, factor=8)
    bo, bi = s[M].split(b, factor=b_factor)
    s[M].reorder(bo, k, bi, eps, co, nu, ci, kk)
    s[V].compute_at(s[M], bo)
    s[M].unroll(ci)
    s[M].vectorize(kk)
    
    # inverse transform
    b, k, eps, nu, kk = s[Y].op.axis
    bo, bi = s[Y].split(b, factor=b_factor)
    s[Y].reorder(bo, k, bi, eps, nu, kk)    
    s[Y].vectorize(kk)
    _ = [s[Y].unroll(x) for x in [eps, nu]]
    
    b, k, eps, nu, kk = s[temp_output_transform].op.axis
    s[temp_output_transform].unroll(eps)
    s[temp_output_transform].unroll(nu)
    s[temp_output_transform].vectorize(kk)
    s[temp_output_transform].compute_at(s[Y], bi)

    last = output_op.output(0)
    n, k, h, w, kk = s[last].op.axis
    ho, hi = s[last].split(h, factor=4)
    wo, wi = s[last].split(w, factor=4)
    s[last].reorder(n, ho, wo, k, hi, wi, kk)
    woo, bi = s[last].split(wo, factor=b_factor)
    bo = s[last].fuse(n, ho, woo)
    s[last].reorder(bo, k, bi, hi, wi, kk)
    s[last].vectorize(kk)
    s[last].parallel(bo)
    s[M].compute_at(s[last], bo)
    s[Y].compute_at(s[last], bo)
    s[last].parallel(bo)

    if conv_op != output_op:
        n, k, h, w, kk = s[output].op.axis
        s[output].vectorize(kk)
        s[conv_op].compute_at(s[output_op], bo)

    return s
    
@generic.schedule_conv2d_winograd_without_filter_transform.register(["cpu"])
def schedule_winograd(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op

    scheduled_ops = []
    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_winograd' in op.tag:
            schedule_winograd_without_filter_transform(s, op, output_op)
        scheduled_ops.append(op)

    traverse(output_op)
    return s
