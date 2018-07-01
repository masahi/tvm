/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_nvptx.cc
 */
#ifdef TVM_LLVM_VERSION

#include "./intrin_rule_llvm.h"
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/api_registry.h>
#include <sstream>

namespace tvm {
namespace codegen {

inline void DispatchExternLibDevice(const TVMArgs& args, TVMRetValue* rv) {
  Expr e = args[0];
  using namespace ir;
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  std::ostringstream intrinsic_name;
  intrinsic_name << "__nv_" << call->name << "f";
  *rv = Call::make(call->type, intrinsic_name.str(), call->args,
                   Call::PureExtern);
}

namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.floor")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.ceil")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.round")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.trunc")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.exp")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.fma")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.log")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.sqrt")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.pow")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.tanh")
.set_body(DispatchExternLibDevice);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
