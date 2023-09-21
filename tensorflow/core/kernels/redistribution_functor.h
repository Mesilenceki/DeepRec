/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_REDISTRIBUTION_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_REDISTRIBUTION_FUNCTOR_H_

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename T>
struct CustomScaleDown {
  void operator()(const Device& d, 
                  typename TTypes<T>::Flat output,
                  typename TTypes<T>::ConstFlat rhs,
                  int partition_id, int partition_num,
                  int offset);
};

template <typename Device, typename T>
struct CustomScaleUp {
  void operator()(const Device& d, 
                  typename TTypes<T>::Flat output,
                  typename TTypes<T>::ConstFlat rhs,
                  int partition_id, int partition_num,
                  int offset);
};

}  // end namespace functor

// template <typename Device>
// Status VariantCopyFn(OpKernelContext* context, 
//                      const Tensor& lhs, 
//                      const Tensor& rhs, Tensor* to);

// template <>
// Status VariantCopyFn<CPUDevice>(OpKernelContext* context, 
//                                 const Tensor& lhs, 
//                                 const Tensor& rhs, Tensor* to);

// template <>
// Status VariantCopyFn<GPUDevice>(OpKernelContext* context, const Tensor& from,
//                                 Tensor* to);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_REDISTRIBUTION_FUNCTOR_H_
