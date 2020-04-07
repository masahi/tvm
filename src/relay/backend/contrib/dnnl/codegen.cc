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
 * \file src/relay/backend/contrib/dnnl/codegen.cc
 * \brief Implementation of DNNL codegen APIs.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>
#include <numeric>

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

// TODO(@zhiics, @comaniac): This is a basic implementation. We should implement
// all utilities and make a base class for users to implement.
class CodegenDNNL : public ExprVisitor, public CodegenCBase {
 public:
  explicit CodegenDNNL(const std::string& id) { this->ext_func_id_ = id; }

  void VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    out_.clear();
    Output output;
    output.name = node->name_hint();
    out_.push_back(output);
    LOG(INFO) << "Visited " << node->name_hint();
  }

  void VisitExpr_(const ConstantNode* cn) final {
    Constant constant = GetRef<Constant>(cn);
    if (visited_.count(constant)) {
      // Note this is for demostration purpose. ConstantNode doesn't necessarily
      // belong to calls. We need to revisit this when tuples come into play.
      out_.push_back(visited_[constant]);
      return;
    }

    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    out_.clear();
    Output output;
    output.name = "const_" + std::to_string(const_idx_++);
    out_.push_back(output);
    visited_[constant] = output;

    runtime::NDArray array = cn->data;
    const auto& shape = array.Shape();
    const DLTensor& dl_tensor = array.ToDLPack()->dl_tensor;

    // Get the number of elements.
    int64_t num_elems = 1;
    for (auto i : shape) num_elems *= i;

    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    CHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);
    // Define a const buffer: float const_0[64] = {1.0, 2.0, ...};
    //
    // Technically, you may need: static float* const_0 = (float*)malloc(4 * 64)
    // to avoid possible stack overflow.
    buf_stream << dtype << " " << output.name << "[" << num_elems << "] = {";
    if (dtype == "float") {
      float* p_flt = static_cast<float*>(dl_tensor.data);
      for (int64_t i = 0; i < num_elems - 1; i++) buf_stream << p_flt[i] << ", ";
      if (num_elems) buf_stream << p_flt[num_elems - 1];
    } else if (dtype == "int") {
      int* p_flt = static_cast<int*>(dl_tensor.data);
      for (int64_t i = 0; i < num_elems - 1; i++) buf_stream << p_flt[i] << ", ";
      if (num_elems) buf_stream << p_flt[num_elems - 1];
    } else {
      LOG(FATAL) << "Only float and int are supported for now.";
    }
    buf_stream << "};";
    ext_func_body.insert(ext_func_body.begin(), buf_stream.str());
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    // Do nothing
  }

  void VisitExpr_(const CallNode* call) final {
    struct GenerateBodyOutput {
      std::string decl, buf;
      int out_size = 1;
      std::string out;
    };

    auto generate_body = [=](const CallNode* root_call, const std::string& func_name,
                             const std::vector<std::string>& args,
                             const std::vector<std::string>& fused_func_args) {
      // Make function call with input buffers when visiting arguments
      bool first = true;
      std::ostringstream decl_stream;
      decl_stream << "(";
      for (size_t i = 0; i < root_call->args.size(); ++i) {
        VisitExpr(root_call->args[i]);
        for (auto out : out_) {
          if (!first) {
            decl_stream << ", ";
          }
          first = false;
          decl_stream << out.name;
        }
      }

      for (auto arg_name : fused_func_args) {
        decl_stream << ", " << arg_name;
      }

      // Analyze the output buffer
      auto type_node = root_call->checked_type().as<TensorTypeNode>();
      CHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
          << "Only support single output tensor with float type";

      auto out_shape = GetShape(root_call->checked_type());

      GenerateBodyOutput ret;
      ret.out = "buf_" + std::to_string(buf_idx_++);
      ret.out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());

      this->PrintIndents();

      std::ostringstream buf_stream;
      buf_stream << "float* " << ret.out << " = (float*)std::malloc(4 * " << ret.out_size << ");";
      ret.buf = buf_stream.str();


      decl_stream << ", " << ret.out;
      // Attach attribute arguments
      for (size_t i = 0; i < args.size(); ++i) {
        decl_stream << ", " << args[i];
      }
      decl_stream << ");";
      ret.decl = func_name + decl_stream.str();

      return ret;
    };


    GenerateBodyOutput ret;
    if (auto conv_call = DetectFusedConv2DBiasReLU(call)) {
      ret = generate_body(conv_call, "dnnl_fused_conv2d_bias_relu",
                          FusedConv2dBiasReLU(conv_call), ext_fused_func_args_);
    } else if (IsOp(call, "nn.conv2d")) {
      ret = generate_body(call, "dnnl_conv2d", Conv2d(call), {});
    } else if (IsOp(call, "nn.dense")) {
      ret = generate_body(call, "dnnl_dense", Dense(call), {});
    } else if (IsOp(call, "nn.relu")) {
      ret = generate_body(call, "dnnl_relu", Relu(call), {});
    } else if (IsOp(call, "nn.batch_norm")) {
      ret = generate_body(call, "dnnl_bn", BatchNorm(call), {});
    } else if (IsOp(call, "add")) {
      ret = generate_body(call, "dnnl_add", Add(call), {});
    } else {
      LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    }

    buf_decl_.push_back(ret.buf);
    ext_func_body.push_back(ret.decl);

    auto type_node = call->checked_type().as<TensorTypeNode>();
    CHECK(type_node);

    // Update output buffer
    out_.clear();
    Output output;
    output.name = ret.out;
    output.dtype = GetDtypeString(type_node);
    output.need_copy = true;
    output.size = ret.out_size;
    out_.push_back(output);
    ext_fused_func_args_.clear();
  }

  std::string JIT(void) {
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out_);
  }

 private:
  const CallNode* DetectFusedConv2DBiasReLU(const CallNode* call) {
    if (!IsOp(call, "nn.relu")) return nullptr;
    auto relu_arg = call->args[0];
    const CallNode* add_call = relu_arg.as<CallNode>();
    if (!add_call || !IsOp(add_call, "add")) return nullptr;
    auto add_arg = add_call->args[0];
    const CallNode* conv_call = add_arg.as<CallNode>();
    if (!conv_call || !IsOp(conv_call, "nn.conv2d")) return nullptr;

    VisitExpr(add_call->args[1]);
    CHECK_EQ(ext_fused_func_args_.size(), 0);
    CHECK_EQ(out_.size(), 1) << "Expected only one constant (bias)";
    ext_fused_func_args_.push_back(out_[0].name);
    return conv_call;
  }

  std::vector<std::string> Conv2d(const CallNode* call) {
    std::vector<std::string> args;
    const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
    CHECK(conv2d_attr);

    auto ishape = GetShape(call->args[0]->checked_type());
    auto wshape = GetShape(call->args[1]->checked_type());

    // Args: N, C, H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    // Args: O, G, Ph, Pw, Kh, Kw, Sh, Sw
    args.push_back(std::to_string(wshape[0]));
    args.push_back(std::to_string(conv2d_attr->groups));
    args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
    args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
    args.push_back(std::to_string(wshape[2]));
    args.push_back(std::to_string(wshape[3]));
    args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
    args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

    return args;
  }

  std::vector<std::string> FusedConv2dBiasReLU(const CallNode* call) {
    return Conv2d(call);
  }

  std::vector<std::string> Dense(const CallNode* call) {
    std::vector<std::string> args;
    auto ishape = GetShape(call->args[0]->checked_type());
    auto wshape = GetShape(call->args[1]->checked_type());

    // Args: N, C, O
    args.push_back(std::to_string(ishape[0]));
    args.push_back(std::to_string(ishape[1]));
    args.push_back(std::to_string(wshape[0]));

    return args;
  }

  std::vector<std::string> Relu(const CallNode* call) {
    std::vector<std::string> args;
    auto ishape = GetShape(call->args[0]->checked_type());

    // Args: N, C, H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    return args;
  }

  std::vector<std::string> BatchNorm(const CallNode* call) {
    std::vector<std::string> args;
    const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
    auto ishape = GetShape(call->args[0]->checked_type());

    // Args: N, C, H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    // Args: epsilon
    args.push_back(std::to_string(bn_attr->epsilon));

    return args;
  }

  std::vector<std::string> Add(const CallNode* call) {
    std::vector<std::string> args;
    auto ishape = GetShape(call->args[0]->checked_type());

    // Args: H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    return args;
  }

  /*! \brief The id of the external dnnl ext_func. */
  std::string ext_func_id_{""};
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The index of global constants. */
  int const_idx_ = 0;
  /*! \brief The arguments used by a wrapped function that calls DNNL kernels. */
  Array<Var> ext_func_args_;
  /*! TODO */
  std::vector<std::string> ext_fused_func_args_;
  /*! \brief statement of the function that will be compiled using DNNL kernels. */
  std::vector<std::string> ext_func_body;
  /*! \brief The declaration of intermeidate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The name of the the outputs. */
  std::vector<Output> out_;
  /*! \brief The cached expressions. */
  std::unordered_map<Expr, Output, ObjectHash, ObjectEqual> visited_;
};

/*!
 * \brief The DNNL codegen helper to generate wrapepr function calls of DNNL
 * libraries. The code is a CSourceModule that can be compiled separately and
 * linked together with a DSOModule.
 */
class DNNLModuleCodegen : public CSourceModuleCodegenBase {
 public:
  // Create a corresponding DNNL function for the given relay Function.
  void GenDNNLFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";
    const auto* call = func->body.as<CallNode>();
    CHECK(call) << "DNNL expects a single convolution or dense op";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    CodegenDNNL builder(sid);
    builder.VisitExpr(func->body);
    code_stream_ << builder.JIT();
  }

  /*!
   * \brief The overridden function that will create a CSourceModule. In order
   * to compile the generated C source code, users need to specify the paths to
   * some libraries, including some TVM required and dnnl specific ones. To make
   * linking simpiler, the DNNL kernels are wrapped in a TVM compatible manner
   * and live under tvm/src/runtime/contrib/dnnl folder.
   *
   * \param ref An object ref that could be either a Relay function or module.
   *
   * \return The runtime module that contains C source code.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    // dnnl_kernel file is saved under src/runtime/contrib/dnnl so that we don't
    // expose it to ordinary users. To make export_library use it, users need to
    // pass -I${PATH_TO_TVM}/src/runtime/contrib
    code_stream_ << "#include <dnnl/dnnl_kernel.h>\n";
    code_stream_ << "using namespace tvm::runtime::contrib;\n";
    code_stream_ << "\n";

    if (ref->IsInstance<FunctionNode>()) {
      GenDNNLFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        GenDNNLFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code_stream_.str(), "cc");
  }

 private:
  /*!
   * \brief The code stream that prints the code that will be compiled using
   * external codegen tools.
   */
  std::ostringstream code_stream_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module DNNLCompiler(const ObjectRef& ref) {
  DNNLModuleCodegen dnnl;
  return dnnl.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.dnnl").set_body_typed(DNNLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
