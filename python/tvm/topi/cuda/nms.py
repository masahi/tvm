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
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison
# pylint: disable=bad-continuation, unused-argument
"""Non-maximum suppression operator"""
import tvm
from tvm import te
from tvm.contrib import nvcc
from tvm.contrib.thrust import can_use_thrust, can_use_rocthrust
from tvm.tir import if_then_else
from .sort import argsort, argsort_thrust
from .scan import exclusive_scan
from ..utils import ceil_div
from ..math import cast
from ..transform import reshape
from ..broadcast import greater


def cuda_atomic_add_rule(op):
    if op.dtype == "float32":
        return tvm.tir.call_pure_extern("float32", "atomicAdd", op.args[0], op.args[1])
    if op.dtype == "float64":
        return tvm.tir.call_pure_extern("float64", "atomicAdd", op.args[0], op.args[1])
    if op.dtype == "int32":
        return tvm.tir.call_pure_extern("int32", "atomicAdd", op.args[0], op.args[1])
    raise RuntimeError("only support int32, float32 and float64")


def opencl_atomic_add_rule(op):
    if op.dtype == "int32":
        return tvm.tir.call_pure_extern("int32", "atomic_add", op.args[0], op.args[1])
    raise RuntimeError("only support int32")


tvm.target.intrin.register_intrin_rule("cuda", "atomic_add", cuda_atomic_add_rule, override=True)

tvm.target.intrin.register_intrin_rule(
    "opencl", "atomic_add", opencl_atomic_add_rule, override=True
)


def atomic_add(x, y):
    return tvm.tir.call_intrin(y.dtype, "tir.atomic_add", x, y)


def get_valid_boxes_ir(data, valid_boxes, score_threshold, id_index, score_index):
    """Low level IR to identify bounding boxes given a score threshold.

    Parameters
    ----------
    data : Buffer
        Input data. 3-D Buffer with shape [batch_size, num_anchors, elem_length].

    score_threshold : Buffer or float32
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    valid_boxes: Buffer
        2D Buffer  indicating valid boxes with shape [batch_size, num_anchors].

    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)

    valid_boxes = ib.buffer_ptr(valid_boxes)
    if isinstance(score_threshold, float):
        score_threshold = tvm.tir.FloatImm("float32", score_threshold)
    id_index = tvm.tir.IntImm("int32", id_index)
    score_index = tvm.tir.IntImm("int32", score_index)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(num_anchors, max_threads)
        nthread_by = batch_size
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        ib.scope_attr(by, "thread_extent", nthread_by)
        tid = bx * max_threads + tx

        with ib.if_scope(tid < num_anchors):
            i = by
            j = tid
            score = data[(i * num_anchors + j) * elem_length + score_index]
            with ib.if_scope(
                tvm.tir.all(
                    score > score_threshold,
                    tvm.tir.any(
                        id_index < 0, data[(i * num_anchors + j) * elem_length + id_index] >= 0
                    ),
                )
            ):
                valid_boxes[i * num_anchors + j] = 1
            with ib.else_scope():
                valid_boxes[i * num_anchors + j] = 0
    return ib.get()


def get_valid_counts_ir(data, valid_indices, valid_boxes, out, out_indices):
    """Low level IR to get valid count of bounding boxes
    given a score threshold. Also prepares to move valid boxes to the
    top of input data.

    Parameters
    ----------
    data : Buffer
        Input data. 3-D Buffer with shape [batch_size, num_anchors, elem_length].

    valid_indices: Buffer
        2D Buffer of flag indicating valid data with shape [batch_size, num_anchors].

    Returns
    -------
    out : Buffer
        Sorted valid boxes

    out_indices : Buffer
        Incidices of valid boxes in original data
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    valid_indices = ib.buffer_ptr(valid_indices)
    valid_boxes = ib.buffer_ptr(valid_boxes)

    out = ib.buffer_ptr(out)
    out_indices = ib.buffer_ptr(out_indices)
    one = tvm.tir.const(1, dtype=out.dtype)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = num_anchors // max_threads + 1
    nthread_by = batch_size
    with ib.new_scope():
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        ib.scope_attr(by, "thread_extent", nthread_by)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < num_anchors):
            i = by
            j = tid
            with ib.for_range(0, elem_length) as k:
                out[(i * num_anchors + j) * elem_length + k] = -one
            out_indices[i * num_anchors + j] = -1
    with ib.new_scope():
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        ib.scope_attr(by, "thread_extent", nthread_by)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < num_anchors):
            i = by
            j = tid
            with ib.if_scope(valid_boxes[i, tid] > 0):
                with ib.for_range(0, elem_length) as k:
                    out[(i * num_anchors + valid_indices[i, tid]) * elem_length + k] = data[
                        (i * num_anchors + j) * elem_length + k
                    ]
                out_indices[i * num_anchors + valid_indices[i, tid]] = j
    return ib.get()


def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, elem_length].

    score_threshold : optional, tvm.te.Tensor or float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes.

    out_tensor : tvm.te.Tensor
        Rearranged data tensor.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    valid_boxes_buf = tvm.tir.decl_buffer(
        (batch_size, num_anchors), "int32", "valid_boxes_buf", data_alignment=8
    )
    valid_boxes = te.extern(
        [(batch_size, num_anchors)],
        [data],
        lambda ins, outs: get_valid_boxes_ir(
            ins[0], outs[0], score_threshold, id_index, score_index
        ),
        dtype=["int32"],
        in_buffers=[data_buf],
        out_buffers=[valid_boxes_buf],
        name="get_valid_boxes",
        tag="get_valid_boxes_gpu",
    )

    valid_indices_buf = tvm.tir.decl_buffer(
        (batch_size, num_anchors), "int32", "valid_indices_buf", data_alignment=8
    )

    valid_indices, valid_count = exclusive_scan(valid_boxes, axis=1, return_reduction=True)

    out_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "out_buf", data_alignment=8)
    out_indices_buf = tvm.tir.decl_buffer(
        (batch_size, num_anchors), "int32", "out_buf", data_alignment=8
    )

    out, out_indices = te.extern(
        [data.shape, (batch_size, num_anchors)],
        [data, valid_indices, valid_boxes],
        lambda ins, outs: get_valid_counts_ir(ins[0], ins[1], ins[2], outs[0], outs[1]),
        dtype=["int32", data.dtype],
        in_buffers=[data_buf, valid_indices_buf, valid_boxes_buf],
        out_buffers=[out_buf, out_indices_buf],
        name="get_valid_counts",
        tag="get_valid_counts_gpu",
    )

    return [valid_count, out, out_indices]


def get_boundaries(output, box_idx):
    l = tvm.te.min(
        output[box_idx],
        output[box_idx + 2],
    )
    t = tvm.te.min(
        output[box_idx + 1],
        output[box_idx + 3],
    )
    r = tvm.te.max(
        output[box_idx],
        output[box_idx + 2],
    )
    b = tvm.te.max(
        output[box_idx + 1],
        output[box_idx + 3],
    )
    return l, t, r, b


def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
    """Calculate overlap of two boxes."""
    a_l, a_t, a_r, a_b = get_boundaries(out_tensor, box_a_idx)
    b_l, b_t, b_r, b_b = get_boundaries(out_tensor, box_b_idx)

    # Overlapping width and height
    w = tvm.te.max(0.0, tvm.te.min(a_r, b_r) - tvm.te.max(a_l, b_l))
    h = tvm.te.max(0.0, tvm.te.min(a_b, b_b) - tvm.te.max(a_t, b_t))

    # Overlapping area
    area = h * w

    # total area of the figure formed by box a and box b
    # except for overlapping area
    u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area
    return tvm.tir.Select(u <= 0.0, 0.0, area / u)


def _nms_loop(
    ib,
    batch_size,
    num_anchors,
    top_k,
    iou_threshold,
    valid_count,
    get_max_output_size_func,
    on_new_valid_box_func,
    on_new_invalidated_box_func,
    needs_bbox_check_func,
    calc_overlap_func,
    out_scores,
    num_valid_boxes,
):
    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

    with ib.new_scope():
        nthread_by = batch_size
        nthread_tx = max_threads

        # Some cuda architectures have smaller limit of 32K for cudaDevAttrMaxRegistersPerBlock
        # vs 64K for most GPUs. Since this kernel uses many registers (around 35), the limit will
        # be exceeded with 1024 threads.
        target = tvm.target.Target.current(allow_none=False)
        if target.kind.name == "cuda":
            if nvcc.get_target_compute_version(target) in ["3.2", "5.3", "6.2"]:
                nthread_tx = 512

        by = te.thread_axis("blockIdx.y")
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(by, "thread_extent", nthread_by)
        ib.scope_attr(tx, "thread_extent", nthread_tx)

        num_valid_boxes_local = ib.allocate(
            "int32", (1,), name="num_valid_boxes_local", scope="local"
        )
        num_valid_boxes_local[0] = 0

        def nms_inner_loop(ib, i, j, nkeep):
            # The box j is valid, invalidate other boxes that overlap with j above iou_threshold
            on_new_valid_box_func(ib, tx, num_valid_boxes_local[0], i, j)
            num_valid_boxes_local[0] += 1

            num_iter_per_thread = ceil_div(nkeep - (j + 1), nthread_tx)

            with ib.for_range(0, num_iter_per_thread, name="_k") as _k:
                k = j + 1 + _k * nthread_tx + tx

                with ib.if_scope(
                    tvm.tir.all(
                        k < nkeep,
                        out_scores[i, k] > 0,  # is the box k still valid?
                        needs_bbox_check_func(i, j, k),
                    )
                ):
                    iou = calc_overlap_func(i, j, k)

                    with ib.if_scope(iou >= iou_threshold):
                        # invalidate the box k
                        out_scores[i, k] = -1.0
                        on_new_invalidated_box_func(i, k)

                ib.emit(tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"])))

        i = by

        max_output_size = get_max_output_size_func(i)
        nkeep = if_then_else(tvm.tir.all(top_k > 0, top_k < valid_count[i]), top_k, valid_count[i])

        with ib.if_scope(tvm.tir.all(iou_threshold > 0, valid_count[i] > 0)):
            # Apply nms
            with ib.if_scope(max_output_size > 0):
                # No need to do more iteration if we have already reached max_output_size boxes
                box_idx = ib.allocate("int32", (1,), name="box_idx", scope="local")
                box_idx[0] = 0
                with ib.while_loop(
                    tvm.tir.all(box_idx[0] < nkeep, num_valid_boxes_local[0] < max_output_size)
                ):
                    # Proceed to the inner loop if the box with id box_idx is still valid
                    with ib.if_scope(out_scores[i, box_idx[0]] > -1.0):
                        nms_inner_loop(ib, i, box_idx[0], nkeep)
                    box_idx[0] += 1

            with ib.else_scope():
                with ib.for_range(0, nkeep, name="j") as j:
                    # Proceed to the inner loop if the box j is still valid
                    with ib.if_scope(out_scores[i, j] > -1.0):
                        nms_inner_loop(ib, i, j, nkeep)

            with ib.if_scope(tx + 0 == 0):
                num_valid_boxes[i] = num_valid_boxes_local[0]

    return ib.get()


def nms_ir(
    data,
    sorted_index,
    valid_count,
    indices,
    out_bboxes,
    out_scores,
    out_class_ids,
    out_features,
    box_indices,
    num_valid_boxes,
    max_output_size,
    iou_threshold,
    force_suppress,
    top_k,
    coord_start,
    id_index,
    score_index,
    return_indices,
):
    """Low level IR routing for transform location in multibox_detection operator.

    Parameters
    ----------
    data : Buffer
        Buffer of output boxes with class and score.

    sorted_index : Buffer
        Buffer of output box indexes sorted by score.

    valid_count : Buffer
        Buffer of number of valid output boxes.

    indices : Buffer
        indices in original tensor, with shape [batch_size, num_anchors],
        represents the index of box in original data. It could be the third
        output out_indices of get_valid_counts. The values in the second
        dimension are like the output of arange(num_anchors) if get_valid_counts
        is not used before non_max_suppression.

    out_bboxes : Buffer
        Output buffer, to be filled with sorted box coordinates.

    out_scores : Buffer
        Output buffer, to be filled with sorted scores.

    out_class_ids : Buffer
        Output buffer, to be filled with sorted class ids.

    box_indices : Buffer
        A indices tensor mapping sorted indices to original indices
        This is the first output of NMS when return_indices=True.

    num_valid_boxes : Buffer
        Record the number of boxes that have survived IOU tests.
        This is the second output of NMS when return_indices=True.

    max_output_size : int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : float
        Overlapping(IoU) threshold to suppress object with smaller score.

    force_suppress : boolean
        Whether to suppress all detections regardless of class_id.

    top_k : int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : int
        Start index of the consecutive 4 coordinates.

    id_index : int
        index of the class categories, -1 to disable.

    score_index : optional, int
        Index of the scores/confidence of boxes.

    return_indices : boolean
        Whether to return box indices in input data.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]
    num_features = out_features.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    sorted_index = ib.buffer_ptr(sorted_index)
    valid_count = ib.buffer_ptr(valid_count)
    indices = ib.buffer_ptr(indices)

    # outputs
    out_bboxes = ib.buffer_ptr(out_bboxes)
    out_scores = ib.buffer_ptr(out_scores)
    out_class_ids = ib.buffer_ptr(out_class_ids)
    out_features = ib.buffer_ptr(out_features)
    box_indices = ib.buffer_ptr(box_indices)
    num_valid_boxes = ib.buffer_ptr(num_valid_boxes)

    if isinstance(iou_threshold, float):
        iou_threshold = tvm.tir.FloatImm("float32", iou_threshold)
    top_k = tvm.tir.IntImm("int32", top_k)
    coord_start = tvm.tir.IntImm("int32", coord_start)
    id_index = tvm.tir.IntImm("int32", id_index)
    score_index = tvm.tir.IntImm("int32", score_index)
    force_suppress = tvm.tir.IntImm("int32", 1 if force_suppress else 0)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(num_anchors, max_threads)
        nthread_by = batch_size
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(by, "thread_extent", nthread_by)
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        i = by
        base_src_idx = i * num_anchors * box_data_length
        base_bbox_idx = i * num_anchors * 4
        base_features_idx = i * num_anchors * num_features

        with ib.if_scope(tvm.tir.all(iou_threshold > 0, valid_count[i] > 0)):
            # Reorder output
            nkeep = if_then_else(
                tvm.tir.all(top_k > 0, top_k < valid_count[i]), top_k, valid_count[i]
            )
            j = bx * max_threads + tx
            with ib.if_scope(j < nkeep):
                src_idx = base_src_idx + sorted_index[i * num_anchors + j] * box_data_length
                with ib.for_range(0, 4, kind="unroll") as k:
                    out_bboxes[(base_bbox_idx + j * 4 + k)] = data[src_idx + coord_start + k]
                with ib.for_range(0, num_features, kind="unroll") as k:
                    out_features[(base_features_idx + j * num_features + k)] = data[
                        src_idx + coord_start + 4 + k
                    ]

                out_scores[i * num_anchors + j] = data[src_idx + score_index]

                if id_index >= 0:
                    out_class_ids[i * num_anchors + j] = data[src_idx + id_index]

            with ib.else_scope():
                # Indices > nkeep are discarded
                # Only needed for return_indices = False case
                if return_indices is False:
                    with ib.if_scope(j < num_anchors):
                        with ib.for_range(0, 4, kind="unroll") as k:
                            out_bboxes[(base_bbox_idx + j * 4 + k)] = -1.0
                        with ib.for_range(0, num_features, kind="unroll") as k:
                            out_features[(base_features_idx + j * num_features + k)] = -1.0

                        out_scores[i, j] = -1.0

                        if id_index >= 0:
                            out_class_ids[i, j] = -1.0

            if return_indices:
                with ib.if_scope(j < num_anchors):
                    box_indices[i * num_anchors + j] = -1

        with ib.else_scope():
            # Need to copy all boxes if not using return_indices
            bounds = valid_count[i] if return_indices else num_anchors
            with ib.if_scope(j < bounds):
                src_offset = base_src_idx + j * box_data_length

                with ib.for_range(0, 4, kind="unroll") as k:
                    out_bboxes[base_bbox_idx + j * 4 + k] = data[src_offset + coord_start + k]
                with ib.for_range(0, num_features, kind="unroll") as k:
                    out_features[(base_features_idx + j * num_features + k)] = data[
                        src_offset + coord_start + 4 + k
                    ]
                out_scores[i * num_anchors + j] = data[src_offset + score_index]

                if id_index >= 0:
                    out_class_ids[i * num_anchors + j] = data[src_offset + id_index]

                box_indices[i * num_anchors + j] = j

    if isinstance(max_output_size, int):
        max_output_size = tvm.tir.const(max_output_size)

    def calc_overlap(i, j, k):
        offset_j = j * 4
        offset_k = k * 4
        base_bbox_idx = i * num_anchors * 4
        return calculate_overlap(
            out_bboxes,
            base_bbox_idx + offset_j,
            base_bbox_idx + offset_k,
        )

    def on_new_valid_box(ib, tid, num_current_valid_box, i, j):
        # When return_indices is False, no need to populate box_indices
        if return_indices:
            with ib.if_scope(tid + 0 == 0):
                orig_idx = sorted_index[i * num_anchors + j]
                box_indices[i, num_current_valid_box] = indices[i, orig_idx]

    def on_new_invalidated_box(i, k):
        if return_indices is False and id_index >= 0:
            out_class_ids[i, k] = -1.0

    def needs_bbox_check(i, j, k):
        return tvm.tir.any(
            force_suppress > 0,
            id_index < 0,
            out_class_ids[i, k] == out_class_ids[i, j],
        )

    return _nms_loop(
        ib,
        batch_size,
        num_anchors,
        top_k,
        iou_threshold,
        valid_count,
        lambda _: max_output_size,
        on_new_valid_box,
        on_new_invalidated_box,
        needs_bbox_check,
        calc_overlap,
        out_scores,
        num_valid_boxes,
    )


def _fetch_score_ir(data, score, axis):
    """
    Fetch score from data.
    This routine is required for dynamic shape nms.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    score = ib.buffer_ptr(score)
    with ib.if_scope(num_anchors > 0):
        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        nthread_tx = max_threads
        nthread_bx = batch_size * num_anchors // max_threads + 1
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)

        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size * num_anchors):
            score[tid] = data[tid * elem_length + axis]

    return ib.get()


def _dispatch_sort(scores, ret_type="indices"):
    target = tvm.target.Target.current()
    if target and (
        can_use_thrust(target, "tvm.contrib.thrust.sort")
        or can_use_rocthrust(target, "tvm.contrib.thrust.sort")
    ):
        return argsort_thrust(scores, axis=1, is_ascend=False, dtype="int32", ret_type=ret_type)
    else:
        return argsort(scores, axis=1, is_ascend=False, dtype="int32", ret_type=ret_type)


def _get_sorted_indices(data, data_buf, score_index, score_shape):
    """Extract a 1D score tensor from the packed input and do argsort on it."""
    score_buf = tvm.tir.decl_buffer(score_shape, data.dtype, "score_buf", data_alignment=8)
    score_tensor = te.extern(
        [score_shape],
        [data],
        lambda ins, outs: _fetch_score_ir(
            ins[0],
            outs[0],
            score_index,
        ),
        dtype=[data.dtype],
        in_buffers=[data_buf],
        out_buffers=[score_buf],
        name="fetch_score",
        tag="fetch_score",
    )
    return _dispatch_sort(score_tensor)


def _run_nms(
    data,
    data_buf,
    sort_tensor,
    valid_count,
    indices,
    max_output_size,
    iou_threshold,
    force_suppress,
    top_k,
    coord_start,
    id_index,
    score_index,
    return_indices,
):
    """Run NMS using sorted scores."""
    sort_tensor_buf = tvm.tir.decl_buffer(
        sort_tensor.shape, sort_tensor.dtype, "sort_tensor_buf", data_alignment=8
    )

    valid_count_dtype = "int32"
    valid_count_buf = tvm.tir.decl_buffer(
        valid_count.shape, valid_count_dtype, "valid_count_buf", data_alignment=4
    )
    indices_buf = tvm.tir.decl_buffer(indices.shape, indices.dtype, "indices_buf", data_alignment=8)

    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    # Number of extra features per box beyond coords, score, and id.
    num_features = data.shape[2] - 6 if id_index >= 0 else data.shape[2] - 5

    # output shapes
    bbox_shape = (batch_size, num_anchors, 4)
    score_shape = (batch_size, num_anchors)
    class_id_shape = score_shape
    out_features_shape = (batch_size, num_anchors, num_features)
    box_indices_shape = score_shape
    num_valid_boxes_shape = (batch_size, 1)

    return te.extern(
        [
            bbox_shape,
            score_shape,
            class_id_shape,
            out_features_shape,
            box_indices_shape,
            num_valid_boxes_shape,
        ],
        [data, sort_tensor, valid_count, indices],
        lambda ins, outs: nms_ir(
            ins[0],
            ins[1],
            ins[2],
            ins[3],
            outs[0],  # sorted bbox
            outs[1],  # sorted scores
            outs[2],  # sorted class ids
            outs[3],  # sorted box feats
            outs[4],  # box_indices
            outs[5],  # num_valid_boxes
            max_output_size,
            iou_threshold,
            force_suppress,
            top_k,
            coord_start,
            id_index,
            score_index,
            return_indices,
        ),
        dtype=[data.dtype, "float32", "float32", "float32", "int32", "int32"],
        in_buffers=[data_buf, sort_tensor_buf, valid_count_buf, indices_buf],
        name="nms",
        tag="nms",
    )


def _concatenate_outputs(
    out_bboxes,
    out_scores,
    out_class_ids,
    out_features,
    out_shape,
    coord_start,
    score_index,
    id_index,
):
    """Pack the results from NMS into a single 5D or 6D tensor."""
    batch_size = out_bboxes.shape[0]
    num_anchors = out_bboxes.shape[1]
    num_features = out_features.shape[2]

    def ir(out_bboxes, out_scores, out_class_ids, out):
        ib = tvm.tir.ir_builder.create()

        out_bboxes = ib.buffer_ptr(out_bboxes)
        out_scores = ib.buffer_ptr(out_scores)
        out_class_ids = ib.buffer_ptr(out_class_ids)
        out = ib.buffer_ptr(out)

        with ib.if_scope(num_anchors > 0):
            max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
            nthread_tx = max_threads
            nthread_bx = ceil_div(num_anchors, nthread_tx)
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            by = te.thread_axis("blockIdx.y")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            ib.scope_attr(by, "thread_extent", batch_size)

            tid = bx * nthread_tx + tx
            i = by

            with ib.if_scope(tid < num_anchors):
                with ib.for_range(0, 4, kind="unroll") as j:
                    out[i, tid, coord_start + j] = out_bboxes[i, tid, j]
                with ib.for_range(0, num_features, kind="unroll") as j:
                    out[i, tid, coord_start + 4 + j] = out_features[i, tid, j]
                out[i, tid, score_index] = out_scores[i, tid]
                if id_index >= 0:
                    out[i, tid, id_index] = out_class_ids[i, tid]

        return ib.get()

    return te.extern(
        [out_shape],
        [out_bboxes, out_scores, out_class_ids],
        lambda ins, outs: ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=["float32"],
        name="nms_output_concat",
        tag="nms_output_concat",
    )


def non_max_suppression(
    data,
    valid_count,
    indices,
    max_output_size=-1,
    iou_threshold=0.5,
    force_suppress=False,
    top_k=-1,
    coord_start=2,
    score_index=1,
    id_index=0,
    return_indices=True,
    invalid_to_bottom=False,
):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, elem_length].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].
        It could be the second output out_tensor of get_valid_counts.

    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes. It could be the output
        valid_count of get_valid_counts.

    indices : tvm.te.Tensor
        2-D tensor with shape [batch_size, num_anchors], represents
        the index of box in original data. It could be the third
        output out_indices of get_valid_counts. The values in the
        second dimension are like the output of arange(num_anchors)
        if get_valid_counts is not used before non_max_suppression.

    max_output_size : optional, tvm.te.Tensor or int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : optional, tvm.te.Tensor or float
        Non-maximum suppression threshold.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id.

    top_k : optional, int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : required, int
        Start index of the consecutive 4 coordinates.

    score_index : optional, int
        Index of the scores/confidence of boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    return_indices : boolean
        Whether to return box indices in input data.

    invalid_to_bottom : optional, boolean
        Whether to move all valid bounding boxes to the top.

    Returns
    -------
    out : tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, elem_length].

    Example
    --------
    .. code-block:: python

        # An example to use nms
        dshape = (1, 5, 6)
        data = te.placeholder(dshape, name="data")
        valid_count = te.placeholder((dshape[0],), dtype="int32", name="valid_count")
        iou_threshold = 0.7
        force_suppress = True
        top_k = -1
        out = non_max_suppression(data=data, valid_count=valid_count, iou_threshold=iou_threshold,
                                 force_suppress=force_supress, top_k=top_k, return_indices=False)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
        s = topi.generic.schedule_nms(out)
        f = tvm.build(s, [data, valid_count, out], "cuda")
        dev = tvm.gpu(0)
        tvm_data = tvm.nd.array(np_data, dev)
        tvm_valid_count = tvm.nd.array(np_valid_count, dev)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), dev)
        f(tvm_data, tvm_valid_count, tvm_out)
    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)

    sort_tensor = _get_sorted_indices(data, data_buf, score_index, (data.shape[0], data.shape[1]))

    out_bboxes, out_scores, out_class_ids, out_features, box_indices, num_valid_boxes = _run_nms(
        data,
        data_buf,
        sort_tensor,
        valid_count,
        indices,
        max_output_size,
        iou_threshold,
        force_suppress,
        top_k,
        coord_start,
        id_index,
        score_index,
        return_indices,
    )

    if return_indices:
        return [box_indices, num_valid_boxes]

    return _concatenate_outputs(
        out_bboxes,
        out_scores,
        out_class_ids,
        out_features,
        data.shape,
        coord_start,
        score_index,
        id_index,
    )


def _get_valid_box_count(scores, score_threshold):
    batch_classes, num_boxes = scores.shape

    def binary_search(ib, y, out):
        lo = ib.allocate("int32", (1,), name="lo", scope="local")
        hi = ib.allocate("int32", (1,), name="hi", scope="local")

        lo[0] = 0
        hi[0] = num_boxes

        with ib.while_loop(lo[0] < hi[0]):
            mid = (hi[0] + lo[0]) >> 1
            with ib.if_scope(scores[y, mid] > score_threshold):
                lo[0] = mid + 1
            with ib.else_scope():
                hi[0] = mid

        out[y] = lo[0]

    def searchsorted_ir(scores, valid_count):
        ib = tvm.tir.ir_builder.create()
        scores = ib.buffer_ptr(scores)
        valid_count = ib.buffer_ptr(valid_count)

        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

        with ib.new_scope():
            ib.scope_attr(bx, "thread_extent", ceil_div(batch_classes, max_threads))
            ib.scope_attr(tx, "thread_extent", max_threads)
            tid = bx * max_threads + tx

            with ib.if_scope(tid < batch_classes):
                binary_search(ib, tid, valid_count)

        return ib.get()

    scores_buf = tvm.tir.decl_buffer(scores.shape, scores.dtype, "scores_buf", data_alignment=8)

    return te.extern(
        [(batch_classes,)],
        [scores],
        lambda ins, outs: searchsorted_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[scores_buf],
        name="searchsorted",
        tag="searchsorted",
    )


def _all_class_nms_ir(
    boxes,
    sorted_scores,
    sorted_indices,
    valid_count,
    batch_class,
    num_class,
    num_anchors,
    iou_threshold,
    max_output_size_per_class,
    box_indices,
    num_valid_boxes,
):
    ib = tvm.tir.ir_builder.create()
    boxes = ib.buffer_ptr(boxes)
    sorted_scores = ib.buffer_ptr(sorted_scores)
    sorted_indices = ib.buffer_ptr(sorted_indices)
    valid_count = ib.buffer_ptr(valid_count)
    box_indices = ib.buffer_ptr(box_indices)
    num_valid_boxes = ib.buffer_ptr(num_valid_boxes)

    def calc_overlap(i, j, k):
        offset_j = sorted_indices[j] * 4
        offset_k = sorted_indices[k] * 4
        base_bbox_idx = i * num_anchors * 4
        return calculate_overlap(
            boxes,
            base_bbox_idx + offset_j,
            base_bbox_idx + offset_k,
        )

    def on_new_valid_box(ib, tid, num_current_valid_box, i, j):
        with ib.if_scope(tid + 0 == 0):
            box_indices[i, num_current_valid_box] = sorted_indices[i * num_anchors + j]

    def max_output_size(batch_class_index):
        return max_output_size_per_class

    def on_new_invalidated_box(i, k):
        pass

    def needs_bbox_check(i, j, k):
        return tvm.tir.const(True)

    if isinstance(iou_threshold, float):
        iou_threshold = tvm.tir.FloatImm("float32", iou_threshold)
    if isinstance(max_output_size, int):
        max_output_size = tvm.tir.const(max_output_size)

    return _nms_loop(
        ib,
        batch_class,
        num_anchors,
        tvm.tir.IntImm("int32", -1),  # top_k
        iou_threshold,
        valid_count,
        max_output_size,
        on_new_valid_box,
        on_new_invalidated_box,
        needs_bbox_check,
        calc_overlap,
        sorted_scores,
        num_valid_boxes,
    )


def _run_all_class_nms(
    boxes, sorted_scores, sorted_indices, valid_count, max_output_size_per_class, iou_threshold
):
    batch, num_boxes, _ = boxes.shape
    batch_class = sorted_scores.shape[0]
    num_class = batch_class // batch

    boxes_buf = tvm.tir.decl_buffer(boxes.shape, boxes.dtype, "boxes_buf", data_alignment=8)
    sorted_scores_buf = tvm.tir.decl_buffer(
        sorted_scores.shape, sorted_scores.dtype, "sorted_scores_buf", data_alignment=8
    )
    sorted_indices_buf = tvm.tir.decl_buffer(
        sorted_indices.shape, sorted_indices.dtype, "sorted_indices_buf", data_alignment=8
    )
    valid_count_buf = tvm.tir.decl_buffer(
        valid_count.shape, "int32", "valid_count_buf", data_alignment=4
    )

    return te.extern(
        [(batch_class, num_boxes), (batch_class,)],
        [boxes, sorted_scores, sorted_indices, valid_count],
        lambda ins, outs: _all_class_nms_ir(
            ins[0],  # boxes
            ins[1],  # sorted_scores
            ins[2],  # sorted_indices
            ins[3],  # valid_count
            batch_class,
            num_class,
            num_boxes,
            iou_threshold,
            max_output_size_per_class,
            outs[0],  # box_indices
            outs[1],  # num_valid_boxes
        ),
        dtype=["int32", "int32"],
        in_buffers=[
            boxes_buf,
            sorted_scores_buf,
            sorted_indices_buf,
            valid_count_buf,
        ],
        name="all_class_nms",
        tag="all_class_nms",
    )


def _collect_selected_indices_ir(num_class, selected_indices, num_detections, row_offsets, out):
    batch_classes, num_boxes = selected_indices.shape

    ib = tvm.tir.ir_builder.create()

    selected_indices = ib.buffer_ptr(selected_indices)
    num_detections = ib.buffer_ptr(num_detections)
    row_offsets = ib.buffer_ptr(row_offsets)
    out = ib.buffer_ptr(out)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_ty = max_threads
    nthread_bx = ceil_div(num_boxes, nthread_tx)
    nthread_by = ceil_div(batch_classes, nthread_ty)
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    bx = te.thread_axis("blockIdx.x")
    by = te.thread_axis("blockIdx.y")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(ty, "thread_extent", nthread_ty)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    ib.scope_attr(by, "thread_extent", nthread_by)

    with ib.new_scope():
        idx = bx * nthread_tx + tx
        idy = by * nthread_ty + ty
        batch_id = idy // num_class
        class_id = idy % num_class
        with ib.if_scope(tvm.tir.all(idy < batch_classes, idx < num_boxes)):
            with ib.if_scope(idx < num_detections[idy]):
                out[row_offsets[idy] + idx, 0] = batch_id
                out[row_offsets[idy] + idx, 1] = class_id
                out[row_offsets[idy] + idx, 2] = idx

    return ib.get()


def _collect_selected_indices(num_class, selected_indices, num_detections, row_offsets):
    batch_class, num_boxes = selected_indices.shape

    selected_indices_buf = tvm.tir.decl_buffer(
        selected_indices.shape, selected_indices.dtype, "selected_indices_buf", data_alignment=8
    )
    num_detections_buf = tvm.tir.decl_buffer(
        num_detections.shape, num_detections.dtype, "num_detections_buf", data_alignment=8
    )
    row_offsets_buf = tvm.tir.decl_buffer(
        row_offsets.shape, row_offsets.dtype, "row_offsets_buf", data_alignment=8
    )

    return te.extern(
        [(batch_class * num_boxes, 3)],
        [selected_indices, num_detections, row_offsets],
        lambda ins, outs: _collect_selected_indices_ir(num_class, ins[0], ins[1], ins[2], outs[0]),
        dtype=["int32"],
        in_buffers=[selected_indices_buf, num_detections_buf, row_offsets_buf],
        name="collect_indices",
        tag="collect_indices",
    )


def all_class_non_max_suppression(
    boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
):
    batch, num_class, num_boxes = scores.shape

    scores = reshape(scores, (batch * num_class, num_boxes))
    sorted_scores, sorted_indices = _dispatch_sort(scores, ret_type="both")
    valid_count = _get_valid_box_count(sorted_scores, score_threshold)

    selected_indices, num_detections = _run_all_class_nms(
        boxes, sorted_scores, sorted_indices, valid_count, max_output_boxes_per_class, iou_threshold
    )
    return selected_indices, num_detections

    # row_offsets, num_total_detections = exclusive_scan(num_detections, return_reduction=True)

    # selected_indices = _collect_selected_indices(
    #     num_class, selected_indices, num_detections, row_offsets
    # )
    # return selected_indices, num_total_detections
