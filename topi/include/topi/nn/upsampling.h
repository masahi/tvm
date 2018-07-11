/*!
 *  Copyright (c) 2017 by Contributors
 * \file topi/nn/upsampling.h
 * \brief upsampling op constructors
 */
#ifndef TOPI_NN_UPSAMPLING_H_
#define TOPI_NN_UPSAMPLING_H_

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

#include "topi/image/resize.h"

namespace topi {
namespace nn {
using namespace tvm;
using namespace topi::image;

inline Tensor upsampling_nearest_nchwc(const Tensor& input,
                                       const Array<Expr>& shape,
                                       std::string name = "tensor",
                                      std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(input->shape[0]);
  out_shape.push_back(input->shape[1]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);
  out_shape.push_back(input->shape[4]);

  Expr h_ratio = shape[0] / input->shape[2];
  Expr w_ratio = shape[1] / input->shape[3];

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    Array<Expr> idx;
    idx.push_back(indices[0]);
    idx.push_back(indices[1]);
    idx.push_back(indices[2] / h_ratio);
    idx.push_back(indices[3] / w_ratio);
    idx.push_back(indices[4]);

    return input(idx);
    }, name, tag);
}
 
/*!
* \brief Upsample given tensor to given shape
*
* \param input The input tensor.
* \param shape Output shape to upsample.
* \param layout input layout
* \param mode Angorithm to use (NEAREST_NEIGHBOR / BILINEAR)
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor upsampled to given shape
*/
inline Tensor upsampling(const Tensor& input,
                         const Array<Expr> shape,
                         std::string layout = "NCHW",
                         std::string mode = "NEAREST_NEIGHBOR",
                         std::string name = "tensor",
                         std::string tag = kInjective) {
  if(input.ndim() == 5){
     return upsampling_nearest_nchwc(input, shape);
  }
  return resize(input, shape, layout, false, mode);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_UPSAMPLING_H_
