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
"""Script to prepare test_addone.so"""
import tvm
import numpy as np
from tvm import te
from tvm import relay
import onnx
import os


def prepare_graph_lib(base_path):
    onnx_model = onnx.load('rtmi_shaped.onnx')
    mod, params = relay.frontend.from_onnx(onnx_model, shape={'bgrm': [1, 4, 176, 96]}, freeze_params=True)
    mod = relay.transform.DynamicToStatic()(mod)
    # build a module
    with tvm.transform.PassContext(opt_level=3):
        compiled_lib = relay.build(mod, tvm.target.create("llvm -mcpu=cascadelake"), params=params)
    # export it as a shared library
    # If you are running cross compilation, you can also consider export
    # to tar and invoke host compiler later.
    dylib_path = os.path.join(base_path, "rtmi_lib.so")
    compiled_lib.export_library(dylib_path)


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_graph_lib(os.path.join(curr_path, "lib"))
