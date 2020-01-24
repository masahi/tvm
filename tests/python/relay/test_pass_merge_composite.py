# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for merge composite."""
from tvm import relay
from tvm.relay.testing import run_opt_pass
import numpy as np
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload
from tvm.relay import analysis, expr as _expr
import tvm.relay.testing
import os
import sys
from tvm.contrib import util

def make_add_sub_mul_pattern():
    """Create a pattern to match the following graph.

        add  sub
         \   /
          \ /
          mul
    """
    x = relay.var('x')
    y = relay.var('y')
    add_node = relay.add(x, y)
    sub_node = relay.subtract(x, y)
    mul_node = relay.multiply(add_node, sub_node)
    return mul_node


def make_add_relu_pattern():
    """Create a pattern to match the following graph.

        add
         |
       ReLu
    """
    x = relay.var('x')
    y = relay.var('y')
    add_node = relay.add(x, y)
    r = relay.nn.relu(add_node)
    return r


def test_simple_merge():
    """Test composite function is correctly produced from simple graph.

    We could expect the pattern `make_add_relu_pattern` to be merged
    into a single op `add_relu`.

        a  b
        \ /               a  b
        add    ====>      \ /
         |             add_relu
       ReLu

    """
    pattern_table = {
        "add_sub_mul": make_add_relu_pattern()
    }

    def before():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))
        add_node = relay.add(a, b)
        r = relay.nn.relu(add_node)
        return relay.Function([a, b], r)

    def expected():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))

        # add_relu function
        in_1 = relay.var('in_1', shape=(10, 10))
        in_2 = relay.var('in_2', shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        relu_node = relay.nn.relu(add_node)
        add_relu = relay.Function([in_1, in_2], relu_node)

        # merged function
        r = relay.Call(add_relu, [a, b])
        return relay.Function([a, b], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert relay.analysis.alpha_equal(result, expected)


def test_branch_merge():
    """Test composite function is correctly produced from branching graph.

    We would expect the pattern `make_add_sub_mul_pattern` to be merged
    into a single op `add_sub_mul`.

       a  b  a  b
        \/    \/
        add  sub                       a  b
         \   /                          \/
          \ /                      add_sub_mul
          mul                     c     |
          /  \                     \    |
       c /  c |       ====>        add_sub_mul
       \/   \/                          |
       add  sub                         |
        \   /                         ReLu
         \ /
         mul
          |
          |
        ReLu
    """

    pattern_table = {
        "add_sub_mul": make_add_sub_mul_pattern()
    }

    def before():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))
        c = relay.var('c', shape=(10, 10))
        add_node = relay.add(a, b)
        sub_node = relay.subtract(a, b)
        mul_node = relay.multiply(add_node, sub_node)
        add_node_2 = relay.add(c, mul_node)
        sub_node_2 = relay.subtract(c, mul_node)
        mul_node_2 = relay.multiply(add_node_2, sub_node_2)
        r = relay.nn.relu(mul_node_2)
        return relay.Function([a, b, c], r)

    def expected():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))
        c = relay.var('c', shape=(10, 10))

        # add_sub_mul function
        in_1 = relay.var('in_1', shape=(10, 10))
        in_2 = relay.var('in_2', shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        sub_node = relay.subtract(in_1, in_2)
        mul_node = relay.multiply(add_node, sub_node)
        add_sub_mul = relay.Function([in_1, in_2], mul_node)

        # merged function
        add_sub_mul_1 = relay.Call(add_sub_mul, [a, b])
        add_sub_mul_2 = relay.Call(add_sub_mul, [c, add_sub_mul_1])
        r = relay.nn.relu(add_sub_mul_2)
        return relay.Function([a, b, c], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert relay.analysis.alpha_equal(result, expected)


def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm",
                 ctx=tvm.cpu(), params=None):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    def update_lib(lib):
        test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        source_dir = os.path.join(test_dir, "..", "..", "..")
        contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

        kwargs = {}
        kwargs["options"] = ["-O2", "-std=c++11", "-I" + contrib_path]
        tmp_path = util.tempdir()
        lib_name = 'lib.so'
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        lib = tvm.module.load(lib_path)

        return lib

    def check_vm_result():
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = relay.vm.Executable.load_exec(code, lib)
        vm = relay.vm.VirtualMachine(exe)
        vm.init(ctx)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    def check_graph_runtime_result():
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            json, lib, param = relay.build(mod, target=target, params=params)
        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, ctx=ctx)
        out = rt_mod.get_output(0, out)

        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    # check_vm_result()
    check_graph_runtime_result()


def test_conv_bn_relu_merge():
    def make_pattern():
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        # weight = relay.const(np.zeros((16, 3, 3, 3)))
        # bias = relay.const(np.zeros((16, 1, 1)))
        weight = relay.var("weight")
        bias = relay.var("bias")
        conv = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3, 3),
                               channels=16, padding=(1, 1))
        add = relay.add(conv, bias)
        return relay.nn.relu(add)

    def get_layers(prefix, data, in_channel, out_channel,
                   include_bn=True, include_sigmoid=False):
        weight = relay.var(prefix + "weight")
        bn_gamma = relay.var(prefix + "bn_gamma")
        bn_beta = relay.var(prefix + "bn_beta")
        bn_mmean = relay.var(prefix + "bn_mean")
        bn_mvar = relay.var(prefix + "bn_var")

        layer = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3, 3),
                                channels=out_channel, padding=(1, 1))
        if include_bn:
            bn_output = relay.nn.batch_norm(layer, bn_gamma, bn_beta,
                                            bn_mmean, bn_mvar)
            layer = bn_output[0]
        if include_sigmoid:
            # dummy layer to prevent pattern detection
            layer = relay.sigmoid(layer)
        layer = relay.nn.relu(layer)
        return layer

    def get_net(include_bn=True, include_sigmoid=False):
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        layer1 = get_layers("layer1_", data, 3, 16, include_bn, include_sigmoid)
        layer2 = get_layers("layer2_", layer1, 16, 16, include_bn, include_sigmoid)
        last = layer2
        return relay.Function(relay.analysis.free_vars(last), last)

    def pre_optimize(mod, params):
        remove_bn_pass = transform.Sequential([
            relay.transform.InferType(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
        ])

        if params != {}:
            # This is required for constant folding
            mod["main"] = bind_params_by_name(mod["main"], params)

        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            mod = remove_bn_pass(mod)

        return mod

    def get_partitoned_mod(mod, params):
        mod = pre_optimize(mod, params)
        pattern_table = {
            "dnnl.conv_bias_relu": make_pattern()
        }
        composite_pass = relay.transform.MergeComposite(pattern_table)
        mod["main"] = run_opt_pass(mod["main"], composite_pass)
        return mod

    def get_partitions(mod):
        partitions = []

        def visit_func(expr):
            if isinstance(expr, _expr.Function) and expr != mod["main"]:
                partitions.append(expr)

        analysis.post_order_visit(mod["main"], visit_func)
        return partitions

    def test_detect_pattern(include_bn, include_sigmoid, num_expected_partition):
        net = get_net(include_bn, include_sigmoid)
        mod, params = tvm.relay.testing.create_workload(net)
        mod = get_partitoned_mod(mod, params)
        assert(len(get_partitions(mod)) == num_expected_partition)

    def test_partition():
        # conv + bn + relu -> detection succeed
        test_detect_pattern(True, False, 2)
        # conv + relu -> fail
        test_detect_pattern(False, False, 0)
        # conv + bn + sigmoid + relu -> fail
        test_detect_pattern(True, True, 0)

    def test_partition_mobilenet():
        mod, params = relay.testing.mobilenet.get_workload()
        mod = get_partitoned_mod(mod, params)
        assert(len(get_partitions(mod)) == 27)

    def test_exec(mod, params, ref_mod, ref_params, out_shape):
        ishape = (1, 3, 224, 224)
        i_data = np.random.randn(*ishape).astype(np.float32)
        ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
        ref_res = ref_ex.evaluate()(i_data, **ref_params)

        mod = get_partitoned_mod(mod, params)
        check_result(mod, {"data": i_data},
                     out_shape, ref_res.asnumpy(), tol=1e-5, params=params)

    test_partition()
    test_partition_mobilenet()

    net = get_net()
    mod, params = tvm.relay.testing.create_workload(net)
    ref_mod, ref_params = tvm.relay.testing.create_workload(net)
    test_exec(mod, params, ref_mod, ref_params, (1, 16, 224, 224))

    mod, params = relay.testing.mobilenet.get_workload()
    ref_mod, ref_params = relay.testing.mobilenet.get_workload()
    test_exec(mod, params, ref_mod, ref_params, (1, 1000))


if __name__ == "__main__":
    test_branch_merge()
    test_simple_merge()
    test_conv_bn_relu_merge()
