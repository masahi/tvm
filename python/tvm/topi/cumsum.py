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
"""Cumsum operator"""
from ..tir import decl_buffer, ir_builder
from ..te import extern
from .transform import reshape
from .utils import prod


def cumsum(data, axis=None, dtype=None):
    if axis is None and axis != 0:
        print("reshape", axis)
        axis = 0
        data = reshape(data, (prod(data.shape),))

    if dtype is None:
        dtype = data.dtype

    shape = data.shape

    axis_mul_before = 1
    axis_mul_after = 1
    if axis < 0:
        axis = len(shape) + axis
    for i, value in enumerate(shape, 0):
        if i < axis:
            axis_mul_before *= value
        elif i > axis:
            axis_mul_after *= value

    print(axis_mul_before, axis_mul_after)

    def gen_ir(data_buf, out_buf):
        ib = ir_builder.create()

        data_buf = ib.buffer_ptr(data_buf)
        out_buf = ib.buffer_ptr(out_buf)

        with ib.for_range(0, axis_mul_before) as i:
            with ib.for_range(0, axis_mul_after) as j:
                base_idx = i * shape[axis] * axis_mul_after + j
                out_buf[base_idx] = data_buf[base_idx]
                with ib.for_range(0, shape[axis] - 1) as _k:
                    k = _k + 1
                    cur_idx = base_idx + k * axis_mul_after
                    prev_idx = base_idx + (k - 1) * axis_mul_after
                    out_buf[cur_idx] = out_buf[prev_idx] + data_buf[cur_idx]

        return ib.get()

    out_buf = decl_buffer(shape, dtype, "out_buf")

    return extern(
        [shape],
        [data],
        lambda ins, outs: gen_ir(ins[0], outs[0]),
        dtype=dtype,
        out_buffers=[out_buf],
        name="scatter_nd_generic",
        tag="scatter_nd_generic",
    )
