/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <chrono>

void DeployGraphRuntime(size_t tid) {
  LOG(INFO) << "Starting Thread" << tid << " on " << sched_getcpu() << "...\n";
  // load in the library
  DLContext ctx{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/rtmi_lib.so");
  // create the graph runtime module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, 4, 176, 96}, DLDataType{kDLFloat, 32, 1}, ctx);
  tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, 1, 176, 96}, DLDataType{kDLFloat, 32, 1}, ctx);

  for (int n = 0; n < 1; ++n) {
    for (int c = 0; c < 4; ++c) {
      for (int h = 0; h < 176; ++h) {
        for (int w = 0; w < 96; ++w) {
          static_cast<float*>(x->data)[(n * 4 * 176 * 96) + (c * 176 * 96) + (h * 96) + w] = n * 2 + c;
        }
      }
    }
  }

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (int i = 0; i < 10000; ++i) {
    LOG(INFO) << tid;
    // set the right input
    set_input("bgrm", x);
    // run the code
    run();
    // get the output
    get_output(0, y);
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

int main(void) {
  //putenv("TVM_NUM_THREADS=1");
  #pragma omp parallel for
  for (size_t tid = 0; tid < 8; ++tid) {
    DeployGraphRuntime(tid);
    //while(true) {
    //}
  }
  return 0;
}
