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
"Scan related operators"
import tvm
from tvm import te
from tvm._ffi import get_global_func
from ..transform import expand_dims, squeeze


def exclusive_sum_scan2d_ir(data, output):
    """
    TODO
    """
    num_rows = data.shape[0]
    scan_size = data.shape[1]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    output = ib.buffer_ptr(output)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

    def ceil_div(a, b):
        return tvm.tir.indexdiv(a + b - 1, b)

    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(scan_size, max_threads)
        nthread_by = num_rows
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        ib.scope_attr(by, "thread_extent", nthread_by)
        tid = bx * nthread_tx + tx
        with ib.if_scope(tid == 0):
            output[by, 0] = 0
        with ib.else_scope():
            with ib.if_scope(tid < scan_size):
                output[by, tid] = data[by, tid - 1]

    nthread_tx = max_threads
    nthread_bx = ceil_div(scan_size, max_threads)
    nthread_by = num_rows

    # Up Sweep of prefix sum
    lim = tvm.tir.generic.cast(
        tvm.tir.ceil(tvm.tir.log2(tvm.tir.generic.cast(scan_size, "float64"))), "int64"
    )
    with ib.for_range(0, lim, dtype="int64") as l2_width:
        width = 2 << l2_width

        with ib.new_scope():
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(
                bx,
                "thread_extent",
                tvm.tir.generic.cast(ceil_div(scan_size, max_threads * width), "int32"),
            )
            tid = bx * nthread_tx + tx

            by = te.thread_axis("blockIdx.y")
            ib.scope_attr(by, "thread_extent", nthread_by)
            start = ib.allocate("int64", (1,), name="start", scope="local")
            middle = ib.allocate("int64", (1,), name="middle", scope="local")
            end = ib.allocate("int64", (1,), name="end", scope="local")
            start[0] = width * tid
            with ib.if_scope(start[0] < scan_size):
                middle[0] = start[0] + tvm.tir.indexdiv(width, 2)
                end[0] = tvm.te.min(start[0] + width, scan_size)
                with ib.if_scope(middle[0] < scan_size):
                    output[by * scan_size + end[0] - 1] += output[by * scan_size + middle[0] - 1]

    # Down Sweep of prefix sum
    with ib.for_range(0, lim - 1, dtype="int64") as l2_width:
        width = 2 << (lim - l2_width - 2)

        with ib.new_scope():
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(
                bx,
                "thread_extent",
                tvm.tir.generic.cast(ceil_div(scan_size, max_threads * width), "int32"),
            )
            tid = bx * nthread_tx + tx

            by = te.thread_axis("blockIdx.y")
            ib.scope_attr(by, "thread_extent", nthread_by)
            start = ib.allocate("int64", (1,), name="start", scope="local")
            middle = ib.allocate("int64", (1,), name="middle", scope="local")
            end = ib.allocate("int64", (1,), name="end", scope="local")
            start[0] = width * tid
            with ib.if_scope(tvm.tir.all(start[0] > 0, start[0] < scan_size)):
                middle[0] = start[0] + tvm.tir.indexdiv(width, 2)
                with ib.if_scope(middle[0] < scan_size):
                    output[by * scan_size + middle[0] - 1] += output[by * scan_size + start[0] - 1]

    return ib.get()


def is_thrust_available():
    """
    Test if thrust based scan ops are available.
    """
    return get_global_func("tvm.contrib.thrust.sum_scan", allow_missing=True) is not None


def scan_thrust(data, exclusive=True):
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    output_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "output_buf", data_alignment=8)
    return te.extern(
        [data.shape],
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.thrust.sum_scan", ins[0], outs[0], exclusive
        ),
        dtype=[data.dtype],
        in_buffers=[data_buf],
        out_buffers=[output_buf],
        name="exclusive_sum_scan2d",
        tag="exclusive_sum_scan2d_gpu",
    )


def exclusive_scan(data, axis=-1):
    # TODO(masahi): support other binary associative operators
    ndim = len(data.shape)
    if axis < 0:
        axis += ndim
    assert axis == ndim - 1, "Only support scan on the inner most axis."

    target = tvm.target.Target.current()
    if target and target.kind.name == "cuda" and is_thrust_available():
        return scan_thrust(data, exclusive=True)

    if ndim == 1:
        data = expand_dims(data, axis=0)

    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    output_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "output_buf", data_alignment=8)

    if ndim == 2:
        output = te.extern(
            [data.shape],
            [data],
            lambda ins, outs: exclusive_sum_scan2d_ir(ins[0], outs[0]),
            dtype=[data.dtype],
            in_buffers=[data_buf],
            out_buffers=[output_buf],
            name="exclusive_scan",
            tag="exclusive_scan_gpu",
        )
        if ndim == 1:
            return squeeze(output, 0)
        return output
    else:
        assert False, "Unsupported dimension {}".format(ndim)
