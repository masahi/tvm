import numpy as np
import tvm
from tvm.contrib import graph_runtime as runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list

def get_sym(layout, kernel_layout, channels):
    data = sym.Variable(name="data")
    data = sym.conv2d(data=data, kernel_size=(3,3), channels=channels, padding=(1, 1),
                      layout=layout, kernel_layout=kernel_layout, use_bias=True)
    data = sym.batch_norm(data)
    data = sym.relu(data)
    data = sym.max_pool2d(data=data, pool_size=(2, 2), strides=(2, 2), layout=layout)
    data = sym.upsampling(data=data, scale=2, layout=layout)
    softmax_axis = 1
    if layout == "NHWC":
        softmax_axis = 3
    data = sym.softmax(data=data, axis=softmax_axis)
    return data


def build_and_run(sym, params, data, out_shape, opt_level=4):
    ctx = tvm.cpu(0)
    import logging
    logging.basicConfig(level=logging.DEBUG)
    with nnvm.compiler.build_config(opt_level=opt_level):
        # with tvm.build_config(auto_unroll_max_step=500,
        #                       unroll_explicit=True):
            graph, lib, params = nnvm.compiler.build(sym, "llvm", shape={"data":data.shape}, params=params)
    # for (k, v) in params.items():
    #     print(k, v.shape)
    #print(graph.json())
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out =  module.get_output(0, tvm.nd.empty(out_shape))
    return out.asnumpy()


def test():
    in_channel = 8
    out_channel = 16
    data_shape = (1, in_channel, 128, 128)    
    nchw_sym = get_sym("NCHW", "OIHW", out_channel)
    nchw_sym2 = get_sym("NCHW", "OIHW", out_channel)    
    conv_weight = np.random.uniform(-1, 1, (out_channel, in_channel, 3, 3)).astype(np.float32)
    conv_bias = np.random.uniform(-1, 1, (out_channel)).astype(np.float32)
    nchw_params = {
        "conv2d0_weight" : tvm.nd.array(conv_weight, ctx=tvm.cpu(0)),
        "conv2d0_bias" : tvm.nd.array(conv_bias, ctx=tvm.cpu(0))
    }

    nchw_params2 = {
        "conv2d1_weight" : tvm.nd.array(conv_weight.copy(), ctx=tvm.cpu(0)),
        "conv2d1_bias" : tvm.nd.array(conv_bias.copy(), ctx=tvm.cpu(0))
    }

    data = np.random.uniform(-1, 1, data_shape).astype(np.float32)
    oshape = (1, out_channel, 128, 128)
    #nchw_output = build_and_run(nchw_sym, nchw_params, data, oshape, opt_level=4)
    nchw_output2 = build_and_run(nchw_sym2, nchw_params2, data, oshape, opt_level=3)
    # print(np.max(np.abs(nchw_output - nchw_output2)))

test()
