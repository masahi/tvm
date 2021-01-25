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
# pylint: disable=invalid-name, too-many-locals, too-many-statements
"Scan related operators"
import tvm
from tvm import te
from tvm._ffi import get_global_func
from ..transform import expand_dims, squeeze, transpose, reshape
from ..utils import ceil_div, swap, prod, get_const_int
from ..math import cast
from .. import tag
from .injective import schedule_injective_from_existing


binop_name_to_func = {"sum": tvm.tir.generic.add}


def exclusive_scan_ir(data, output, reduction=None, binop="sum"):
    """Low level IR to do exclusive sum scan along rows of 2D input.

    Parameters
    ----------
    data : Buffer
        Input N-D Buffer. Scan is done over the innermost axis.

    output: Buffer
        A buffer to store the output scan, of the same shape as data

    reduction: Buffer, optional
        (N-1)-D Buffer, to store the sum of each scan axis.

    biop: string, optional
        A string specifying which binary operator to use. Currently only "sum" is supported.
    """

    batch_size = prod(data.shape[:-1])
    scan_axis_size = data.shape[-1]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    output = ib.buffer_ptr(output)

    out_dtype = output.dtype

    if reduction is not None:
        reduction = ib.buffer_ptr(reduction)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

    with ib.if_scope(scan_axis_size == 0):
        with ib.new_scope():
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(bx, "thread_extent", batch_size)
            with ib.if_scope(bx < batch_size):
                if reduction is not None:
                    reduction[bx] = 0
    with ib.else_scope():
        with ib.new_scope():
            nthread_tx = max_threads
            nthread_bx = ceil_div(scan_axis_size, max_threads)
            nthread_by = batch_size
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            by = te.thread_axis("blockIdx.y")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            ib.scope_attr(by, "thread_extent", nthread_by)
            tid = bx * nthread_tx + tx
            with ib.if_scope(tid < scan_axis_size):
                output[by * scan_axis_size + tid] = cast(data[by * scan_axis_size + tid], out_dtype)

        nthread_tx = max_threads
        nthread_bx = ceil_div(scan_axis_size, max_threads)
        nthread_by = batch_size

        # The following algorithm performs parallel exclusive scan
        # Up Sweep of exclusive scan
        lim = tvm.tir.generic.cast(
            tvm.tir.ceil(tvm.tir.log2(tvm.tir.generic.cast(scan_axis_size, "float64"))), "int64"
        )
        op = binop_name_to_func[binop]
        with ib.for_range(0, lim, dtype="int64") as l2_width:
            width = 2 << l2_width

            with ib.new_scope():
                tx = te.thread_axis("threadIdx.x")
                bx = te.thread_axis("blockIdx.x")
                ib.scope_attr(tx, "thread_extent", nthread_tx)
                ib.scope_attr(
                    bx,
                    "thread_extent",
                    tvm.tir.generic.cast(ceil_div(scan_axis_size, max_threads * width), "int32"),
                )
                tid = bx * nthread_tx + tx

                by = te.thread_axis("blockIdx.y")
                ib.scope_attr(by, "thread_extent", nthread_by)
                start = ib.allocate("int64", (1,), name="start", scope="local")
                middle = ib.allocate("int64", (1,), name="middle", scope="local")
                end = ib.allocate("int64", (1,), name="end", scope="local")
                start[0] = width * tid
                with ib.if_scope(start[0] < scan_axis_size):
                    middle[0] = start[0] + tvm.tir.indexdiv(width, 2)
                    end[0] = tvm.te.min(start[0] + width, scan_axis_size)
                    with ib.if_scope(middle[0] < scan_axis_size):
                        output[by * scan_axis_size + end[0] - 1] = op(
                            output[by * scan_axis_size + end[0] - 1],
                            output[by * scan_axis_size + middle[0] - 1],
                        )

        # Down Sweep of exclusive scan
        with ib.new_scope():
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(bx, "thread_extent", batch_size)
            with ib.if_scope(bx < batch_size):
                if reduction is not None:
                    reduction[bx] = output[(bx + 1) * scan_axis_size - 1]
                output[(bx + 1) * scan_axis_size - 1] = cast(0, out_dtype)

        with ib.for_range(0, lim, dtype="int64") as l2_width:
            width = 2 << (lim - l2_width - 1)

            with ib.new_scope():
                tx = te.thread_axis("threadIdx.x")
                bx = te.thread_axis("blockIdx.x")
                ib.scope_attr(tx, "thread_extent", nthread_tx)
                ib.scope_attr(
                    bx,
                    "thread_extent",
                    tvm.tir.generic.cast(ceil_div(scan_axis_size, max_threads * width), "int32"),
                )
                tid = bx * nthread_tx + tx

                by = te.thread_axis("blockIdx.y")
                ib.scope_attr(by, "thread_extent", nthread_by)
                start = ib.allocate("int64", (1,), name="start", scope="local")
                middle = ib.allocate("int64", (1,), name="middle", scope="local")
                end = ib.allocate("int64", (1,), name="end", scope="local")
                tmp = ib.allocate(out_dtype, (1,), name="end", scope="local")
                start[0] = width * tid
                with ib.if_scope(tvm.tir.all(start[0] < scan_axis_size)):
                    middle[0] = start[0] + tvm.tir.indexdiv(width, 2)
                    end[0] = tvm.tir.min(start[0] + width, scan_axis_size)
                    with ib.if_scope(middle[0] < scan_axis_size):
                        tmp[0] = output[by * scan_axis_size + middle[0] - 1]
                        output[by * scan_axis_size + middle[0] - 1] = output[
                            by * scan_axis_size + end[0] - 1
                        ]
                        output[by * scan_axis_size + end[0] - 1] = op(
                            output[by * scan_axis_size + end[0] - 1], tmp[0]
                        )
    return ib.get()


def get_reduction_from_exclusive_scan(data, ex_scan_output, binop="sum"):
    """Return the sum of the last element of data and the exclusive scan output.
    The is the reduction of data along each row (for 2-D case).

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data of any shape

    ex_scan_output : tvm.te.Tensor
        The output of exclusive scan on data

    biop: string, optional
        A string specifying which binary operator to use. Currently only "sum" is supported.

    Returns
    -------
    reduction : tvm.te.Tensor
        (N-1)-D tensor storing the reduction of each scan axis.
    """
    ndim = len(data.shape)
    if ndim == 1:
        data = expand_dims(data, axis=0)
        ex_scan_output = expand_dims(ex_scan_output, axis=0)

    def ir(data, data_ex_scan, reduction):
        batch_size = prod(data.shape[:-1])
        scan_axis_size = data.shape[-1]

        ib = tvm.tir.ir_builder.create()

        data = ib.buffer_ptr(data)
        data_ex_scan = ib.buffer_ptr(data_ex_scan)
        reduction = ib.buffer_ptr(reduction)

        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        with ib.new_scope():
            nthread_tx = max_threads
            nthread_bx = ceil_div(batch_size, max_threads)
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            tid = bx * max_threads + tx
            with ib.if_scope(tid < batch_size):
                with ib.if_scope(scan_axis_size > 0):
                    reduction[tid] = binop_name_to_func[binop](
                        data_ex_scan[tid * scan_axis_size + scan_axis_size - 1],
                        data[tid, scan_axis_size - 1],
                    )
                with ib.else_scope():
                    reduction[tid] = 0

        return ib.get()

    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "valid_indices_buf", data_alignment=8)
    ex_scan_output_buf = tvm.tir.decl_buffer(
        ex_scan_output.shape, ex_scan_output.dtype, "ex_scan_output_buf", data_alignment=8
    )

    reduction = te.extern(
        [data.shape[:-1]],
        [data, ex_scan_output],
        lambda ins, outs: ir(ins[0], ins[1], outs[0]),
        dtype=[ex_scan_output.dtype],
        in_buffers=[data_buf, ex_scan_output_buf],
        name="ex_scan_reduction",
        tag="ex_scan_reduction_gpu",
    )

    if ndim == 1:
        return squeeze(reduction, 0)

    return reduction


def is_thrust_available():
    """Test if thrust based scan ops are available."""
    return get_global_func("tvm.contrib.thrust.sum_scan", allow_missing=True) is not None


def scan_thrust(data, output_dtype, exclusive=True, return_reduction=False, binop="sum"):
    """Do exclusive or inclusive scan on 1D or multidimensional input, using thrust.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data of any shape. The scan is done over the innermost axis.

    output_dtype: string
        The dtype of the output scan tensor.

    exclusive: bool, optional
        Whether or not do exclusive or inclusive scan.

    return_reduction: bool, optional
        Whether or not return a (N-1)-D tensor storing the reduction of each scan axis.
        Reductions are computed as part of the upsweep pass, so there is no extra cost.
        If False, reductions are ignored. It must be False when exclusive is False.

    biop: string, optional
        A string specifying which binary operator to use. Currently only "sum" is supported.

    Returns
    -------
    output : tvm.te.Tensor
        A N-D tensor of the same rank N and shape as the input data.

    reduction : tvm.te.Tensor, optional
        (N-1)-D tensor storing the reduction of each scan axis.
        Returned if return_reduction is True.
    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    output_buf = tvm.tir.decl_buffer(data.shape, output_dtype, "output_buf", data_alignment=8)
    binop_to_thrust_func_name = {"sum": "tvm.contrib.thrust.sum_scan"}
    output = te.extern(
        [data.shape],
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            binop_to_thrust_func_name[binop], ins[0], outs[0], exclusive
        ),
        dtype=[output_dtype],
        in_buffers=[data_buf],
        out_buffers=[output_buf],
        name="exclusive_scan_thrust",
        tag="exclusive_scan_thrust_gpu",
    )

    if return_reduction:
        assert exclusive, "return_reduction should be False for inclusive scan"
        reduction = get_reduction_from_exclusive_scan(data, output, binop)
        return output, reduction

    return output


def exclusive_scan(data, axis=-1, return_reduction=False, output_dtype=None, binop="sum"):
    """Do exclusive scan on 1D or multidimensional input.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data of any shape.

    axis: int, optional
        The axis to do scan on. By default, scan is done on the innermost axis.

    return_reduction: bool, optional
        Whether or not return a tensor storing the reduction over each scan axis.
        If the input rank is N, this tensor is of rank N - 1.
        Reductions are computed as part of the upsweep pass, so there is no extra cost.
        If False, reductions are ignored.

    output_dtype: string, optional
        The dtype of the output scan tensor. If not provided, the dtype of the input is used.

    biop: string, optional
        A string specifying which binary operator to use. Currently only "sum" is supported.

    Returns
    -------
    output : tvm.te.Tensor
        A N-D tensor of the same rank N and shape as the input data.

    reduction : tvm.te.Tensor, optional
        (N-1)-D tensor storing the reduction of each scan axis.
        Returned if return_reduction is True.
    """

    def do_scan(data, output_dtype):
        target = tvm.target.Target.current()
        if target and target.kind.name == "cuda" and is_thrust_available():
            return scan_thrust(
                data, output_dtype, exclusive=True, return_reduction=return_reduction, binop=binop
            )

        if ndim == 1:
            # TIR exclusive scan accepts only 2D or higher-rank inputs.
            data = expand_dims(data, axis=0)

        data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
        output_buf = tvm.tir.decl_buffer(data.shape, output_dtype, "output_buf", data_alignment=8)

        if return_reduction:
            output, reduction = te.extern(
                [data.shape, data.shape[:-1]],
                [data],
                lambda ins, outs: exclusive_scan_ir(ins[0], outs[0], outs[1], binop=binop),
                dtype=[data.dtype, output_dtype],
                in_buffers=[data_buf],
                name="exclusive_scan",
                tag="exclusive_scan_gpu",
            )
        else:
            output = te.extern(
                [data.shape],
                [data],
                lambda ins, outs: exclusive_scan_ir(ins[0], outs[0], binop=binop),
                dtype=[output_dtype],
                in_buffers=[data_buf],
                out_buffers=[output_buf],
                name="exclusive_scan",
                tag="exclusive_scan_gpu",
            )
            reduction = None

        if ndim == 1:
            output = squeeze(output, 0)
            if return_reduction:
                reduction = squeeze(reduction, 0)

        if return_reduction:
            return output, reduction

        return output

    if output_dtype is None or output_dtype == "":
        output_dtype = data.dtype

    ndim = len(data.shape)
    if axis < 0:
        axis += ndim

    # If scan axis is not the innermost one, swap the scan and the innermost axes
    # Scan is always done on the innermost axis, for performance reason.
    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    if return_reduction:
        output, reduction = do_scan(data, output_dtype)
    else:
        output = do_scan(data, output_dtype)

    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        output = transpose(output, axes)

    if return_reduction:
        return output, reduction

    return output


def inclusive_scan(data, axis=-1, output_dtype=None, binop="sum"):
    """Do inclusive scan on 1D or multidimensional input.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data of any shape.

    axis: int, optional
        The axis to do scan on. By default, scan is done on the innermost axis.

    output_dtype: string, optional
        The dtype of the output scan tensor. If not provided, the dtype of the input is used.

    biop: string, optional
        A string specifying which binary operator to use. Currently only "sum" is supported.

    Returns
    -------
    output : tvm.te.Tensor
        A N-D tensor of the same rank N as the input data.
    """
    ex_scan = exclusive_scan(data, axis, output_dtype=output_dtype, binop=binop)

    if output_dtype is not None and data.dtype != output_dtype and output_dtype != "":
        data = cast(data, output_dtype)

    return binop_name_to_func[binop](data, ex_scan)


def schedule_scan(outs):
    """Schedule for scan operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of scan
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        if tag.is_injective(op.tag):
            schedule_injective_from_existing(s, op.output(0))
        for tensor in op.input_tensors:
            if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                traverse(tensor.op)
        scheduled_ops.append(op)

    for out in outs:
        traverse(out.op)
    return s


def cumsum(data, axis=None, dtype=None):
    """Numpy style cumsum op. Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    axis : int, optional
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    dtype : string, optional
        Type of the returned array and of the accumulator in which the elements are summed.
        If dtype is not specified, it defaults to the dtype of data.

    Returns
    -------
    result : tvm.te.Tensor
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.
    """
    if axis is None:
        axis = 0
        data = reshape(data, (prod(data.shape),))
    axis = get_const_int(axis)
    return inclusive_scan(data, axis, output_dtype=dtype, binop="sum")
