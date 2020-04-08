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
#include <numeric>
#include <sstream>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

const CallNode* GetRootConv2DCall(const CallNode* relu_call) {
  CHECK(relu_call && IsOp(relu_call, "nn.relu"));
  const auto relu_arg = relu_call->args[0];
  const CallNode* add_call = relu_arg.as<CallNode>();
  CHECK(add_call && IsOp(add_call, "add"));
  const auto add_arg = add_call->args[0];
  const CallNode* conv_call = add_arg.as<CallNode>();
  CHECK(conv_call && IsOp(conv_call, "nn.conv2d"));
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
  }

  void VisitExpr_(const ConstantNode* cn) final {
    Constant constant = GetRef<Constant>(cn);
    if (visited_.count(constant)) {
      out_.push_back(visited_[constant]);
      return;
    }

    out_.clear();
    Output output;
    output.name = "const_" + std::to_string(const_idx_++);
    output.dtype = "float";
    out_.push_back(output);
    visited_[constant] = output;

    runtime::NDArray array = cn->data;

    // Get the number of elements.
    int64_t num_elems = 1;
    for (auto i : array.Shape()) num_elems *= i;

    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    CHECK(type_node);
    CHECK_EQ(GetDtypeString(type_node), "float") << "Only float is supported for now.";

    std::ostringstream buf_stream;
    buf_stream << "float* " << output.name << " = (float*)std::malloc(4 * " << num_elems << ");\n";
    const float* ptr = static_cast<float*>(array.ToDLPack()->dl_tensor.data);
    for (int64_t i = 0; i < num_elems; i++) {
      buf_stream << "  " << output.name << "[" << i << "] = " << ptr[i] << ";\n";
    }

    ext_func_body.insert(ext_func_body.begin(), buf_stream.str());
  }

  void VisitExpr_(const CallNode* call) final {
    GenerateBodyOutput ret;
    if (const auto* func = call->op.as<FunctionNode>()) {
      ret = GenerateCompositeFunctionCall(func, call);
    } else {
      ret = GenerateOpCall(call);
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
  }

  std::string JIT(void) {
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out_);
  }

 private:
  struct GenerateBodyOutput {
    std::string decl, buf;
    int out_size = 1;
    std::string out;
  };

  std::vector<std::string> GetArgumentNames(const CallNode* call) {
    std::vector<std::string> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]);
      for (auto out : out_) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  GenerateBodyOutput GenerateOpCall(const CallNode* call) {
    const auto* op_node = call->op.as<OpNode>();
    CHECK(op_node) << "OpNode expected, got something else";

    using ArgFunType = std::function<std::vector<std::string>(const CallNode*)>;
    const std::map<std::string, std::pair<std::string, ArgFunType>> op_map = {
        {"nn.conv2d", {"dnnl_conv2d", Conv2d}},
        {"nn.dense", {"dnnl_dense", Dense}},
        {"nn.relu", {"dnnl_relu", Relu}},
        {"nn.batch_norm", {"dnnl_bn", BatchNorm}},
        {"add", {"dnnl_add", Add}},
    };

    const auto op_name = GetRef<Op>(op_node)->name;
    const auto iter = op_map.find(op_name);
    if (iter != op_map.end()) {
      return GenerateBody(call, iter->second.first, iter->second.second(call));
    }

    LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    return GenerateBodyOutput{};
  }

  GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                   const CallNode* caller) {
    const auto comp_name = callee->GetAttr<tir::StringImm>(attr::kComposite);
    CHECK(comp_name.defined()) << "Only functions with composite attribute supported";
    if (comp_name->value == "dnnl.conv_bias_relu") {
      const auto* conv_call = GetRootConv2DCall(callee->body.as<CallNode>());
      return GenerateBody(conv_call, "dnnl_fused_conv2d_bias_relu", GetArgumentNames(caller),
                          Conv2d(conv_call));
    }
    LOG(FATAL) << "Unknown composite function:" << comp_name;
    return GenerateBodyOutput{};
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& attribute_args) {
    return GenerateBody(root_call, func_name, GetArgumentNames(root_call), attribute_args);
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& func_args,
                                  const std::vector<std::string>& attribute_args) {
    // Make function call with input buffers when visiting arguments
    CHECK_GT(func_args.size(), 0);
    std::ostringstream decl_stream;
    decl_stream << "(" << func_args[0];
    for (size_t i = 1; i < func_args.size(); ++i) {
      decl_stream << ", " << func_args[i];
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
    for (size_t i = 0; i < attribute_args.size(); ++i) {
      decl_stream << ", " << attribute_args[i];
    }
    decl_stream << ");";
    ret.decl = func_name + decl_stream.str();
    return ret;
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
