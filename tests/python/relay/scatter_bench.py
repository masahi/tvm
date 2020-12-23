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
""" Support level3 operator test cases.
"""
import numpy as np
import tvm
from tvm import relay
import tvm.testing
from tvm.contrib import graph_runtime


@tvm.testing.uses_gpu
def test_scatter():
    def ref_scatter(data, indices, updates, axis=0):
        idx = np.indices(indices.shape).reshape(indices.ndim, -1)

        updated_idx = np.copy(idx)
        indices = indices.reshape(-1)
        for i in range(len(indices)):
            updated_idx[axis, i] = indices[i]
        scattered = np.copy(data)
        scattered[tuple(updated_idx)] = updates[tuple(idx)]
        return scattered

    def verify_scatter(dshape, ishape, axis=0):
        d = relay.var("d", relay.TensorType(dshape, "float32"))
        i = relay.var("i", relay.TensorType(ishape, "int64"))
        u = relay.var("u", relay.TensorType(ishape, "float32"))
        z = relay.op.scatter(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np = np.random.uniform(size=dshape).astype("float32")
        updates_np = np.random.uniform(size=ishape).astype("float32")
        indices_np = np.random.randint(-dshape[axis], dshape[axis] - 1, ishape).astype("int64")

        ref_res = ref_scatter(data_np, indices_np, updates_np, axis)

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph"]:
                if target == "cuda":
                    intrp = relay.create_executor(kind, ctx=ctx, target=target)
                    op_res = intrp.evaluate(func)(data_np, indices_np, updates_np)
                    tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

                    mod = tvm.ir.IRModule.from_expr(func)
                    lib = relay.build(mod, target=target)
                    module = graph_runtime.GraphModule(lib["default"](ctx))
                    module.set_input("d", tvm.nd.array(data_np))
                    module.set_input("i", tvm.nd.array(indices_np))
                    module.set_input("u", tvm.nd.array(updates_np))
                    ftimer = module.module.time_evaluator("run", ctx, repeat=100)
                    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
                    print("size %d, elapsed ms: %f" % ( dshape[0], prof_res.mean()))


    np.random.seed(123)
    for size in [5000, 10000, 25000, 50000, 100000, 500000, 1000000]:
        verify_scatter((size,), (size,), 0)


if __name__ == "__main__":
    test_scatter()
