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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/redistribution_functor.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <>
struct CustomScaleDown<CPUDevice, string> {
  void operator()(const CPUDevice& d, 
                  typename TTypes<tstring>::Flat output,
                  typename TTypes<tstring>::ConstFlat rhs,
                  int partition_id, int partition_num) {
    if (output.dimension(0) == 1) {
      output.data()->resize(rhs.data()->size());
      auto work = [&output, &rhs](int64 start, int64 end) {
        memmove(const_cast<char*>(output.data()->data()) + start,
                rhs.data()->data() + start, end - start);
      };
      d.parallelFor(rhs.data()->size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto offset = rhs.dimension(0);
      auto work = [&output, &rhs](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i].resize(rhs.data()[i].size());
          memmove(const_cast<char*>(output.data()[i].data()),
                  rhs.data()[i].data(), rhs.data()[i].size());
        }
      };
      int64 estimated_string_size;
      if (rhs.size() > 0) {
        // first element of the tensor seems as good a guess as any of the sizes
        // of the strings contained within...
        estimated_string_size =
            std::max(rhs.data()[0].size(), sizeof(tstring));
      } else {
        estimated_string_size = sizeof(tstring);
      }
      d.parallelFor(
          offset,
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);

      //offset
      auto copy_work = [&output, &rhs, offset](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[offset + i].resize(rhs.data()[i].size());
          memmove(const_cast<char*>(output.data()[offset + i].data()),
                  rhs.data()[i].data(), rhs.data()[i].size());
        }
      };
      if (rhs.size() > 0) {
        // first element of the tensor seems as good a guess as any of the sizes
        // of the strings contained within...
        estimated_string_size =
            std::max(rhs.data()[0].size(), sizeof(tstring));
      } else {
        estimated_string_size = sizeof(tstring);
      }
      d.parallelFor(
          rhs.dimension(0),
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          copy_work);
    }
  }
};

template<typename T>
struct CustomScaleDown<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
                  typename TTypes<T>::Flat output,
                  typename TTypes<T>::ConstFlat rhs,
                  int partition_id, int partition_num) {
    if (output.dimension(0) == 1) {
      auto work = [&output, &rhs](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i] = rhs.data()[i];
        }
      };
      d.parallelFor(rhs.size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto size = output.dimension(0);
      int64 offset = partition_id * size;

      LOG(INFO) << "size is: " << size << " offset is: " << offset;

      auto work = [&output, &rhs, offset](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i] = rhs.data()[i+offset];
        }
      };
      int64 estimated_string_size = sizeof(T);
      d.parallelFor(
          size,
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);

    }
  }
};

template <>
struct CustomScaleUp<CPUDevice, string> {
  void operator()(const CPUDevice& d, 
                  typename TTypes<tstring>::Flat output,
                  typename TTypes<tstring>::ConstFlat rhs,
                  int partition_id, int partition_num) {
    if (output.dimension(0) == 1) {
      output.data()->resize(rhs.data()->size());
      auto work = [&output, &rhs](int64 start, int64 end) {
        memmove(const_cast<char*>(output.data()->data()) + start,
                rhs.data()->data() + start, end - start);
      };
      d.parallelFor(rhs.data()->size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto size = rhs.dimension(0);
      int64 offset = partition_id * size;
      auto work = [&output, &rhs, &offset](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i].resize(rhs.data()[i].size());
          memmove(const_cast<char*>(output.data()[i].data()),
                  rhs.data()[i+offset].data(), rhs.data()[i+offset].size());
        }
      };
      int64 estimated_string_size;
      if (rhs.size() > 0) {
        // first element of the tensor seems as good a guess as any of the sizes
        // of the strings contained within...
        estimated_string_size =
            std::max(rhs.data()[0].size(), sizeof(tstring));
      } else {
        estimated_string_size = sizeof(tstring);
      }
      d.parallelFor(
          size,
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);
    }
  }
};

template<typename T>
struct CustomScaleUp<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
                  typename TTypes<T>::Flat output,
                  typename TTypes<T>::ConstFlat rhs,
                  int partition_id, int partition_num) {
    if (output.dimension(0) == 1) {
      auto work = [&output, &rhs](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i] = rhs.data()[i];
        }
      };
      d.parallelFor(rhs.size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto size = output.dimension(0);
      int64 offset = partition_id * size;

      LOG(INFO) << "size is: " << size << " offset is: " << offset;

      auto work = [&output, &rhs, offset](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          output.data()[i] = rhs.data()[i+offset];
        }
      };
      int64 estimated_string_size = sizeof(T);
      d.parallelFor(
          size,
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);

    }
  }
};

}  // namespace functor

#define CPU_DENSE_ASSIGN(type)                                       \
  template struct functor::CustomScaleDown<CPUDevice, type>;         \
  template struct functor::CustomScaleUp<CPUDevice, type>;


TF_CALL_REAL_NUMBER_TYPES(CPU_DENSE_ASSIGN);

#undef CPU_DENSE_ASSIGN

// #define CPU_DENSE_COPY(T)                                                \
//   case DataTypeToEnum<T>::value: {                                       \
//     functor::CustomDenseUpdate<CPUDevice, T> copy_functor_;              \
//     copy_functor_(context->eigen_device<CPUDevice>(),                    \
//                   output->flat<T>(),                                     \
//                   lhs.flat<T>(),                                         \    
//                   rhs.flat<T>());                                        \
//     break;                                                               \
//   }

// #define INSTANTIATE_GET_VARIANT_COPY_FN(DEVICE, TYPE_CALLER, TYPE_DENSE_COPY) \
//   template <>                                                                 \
//   Status VariantCopyFn<DEVICE>(OpKernelContext * context,                     \
//                                const Tensor& lhs,                             \
//                                const Tensor& rhs,                            \
//                                Tensor* to) {                                  \
//     PersistentTensor tmp;                                                     \
//     Tensor* tensor;                                                           \
//     AllocatorAttributes attr;                                                 \
//     attr.set_gpu_compatible(true);                                            \
//     attr.set_nic_compatible(true);                                            \
//     TF_RETURN_IF_ERROR(context->allocate_persistent(                          \
//         lhs.dtype(), lhs.shape(), &tmp, &tensor, attr));                    \
//     switch (from.dtype()) {                                                   \
//       TYPE_CALLER(TYPE_DENSE_COPY);                                           \
//       default:                                                                \
//         return errors::InvalidArgument(                                       \
//             "VariantCopyFn: Could not perform a deep copy of variant "        \
//             "element of type: ",                                              \
//             DataTypeString(from.dtype()),                                     \
//             " using device: ", context->device()->name());                    \
//     }                                                                         \
//     *to = *tensor;                                                            \
//     return Status::OK();                                                      \
//   }

// INSTANTIATE_GET_VARIANT_COPY_FN(CPUDevice, TF_CALL_ALL_TYPES, CPU_DENSE_COPY);

// #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// #define GPU_DENSE_COPY(T)                                                \
//   case DataTypeToEnum<T>::value: {                                       \
//     functor::DenseUpdate<GPUDevice, T, ASSIGN> copy_functor_;            \
//     copy_functor_(context->eigen_device<GPUDevice>(), tensor->flat<T>(), \
//                   from.flat<T>());                                       \
//     break;                                                               \
//   }
// #define TF_CALL_GPU_AND_ADDITIONAL_TYPES(T) \
//   TF_CALL_GPU_ALL_TYPES(T);                 \
//   TF_CALL_int32(T);                         \
//   TF_CALL_int64(T);
// INSTANTIATE_GET_VARIANT_COPY_FN(GPUDevice, TF_CALL_GPU_AND_ADDITIONAL_TYPES,
//                                 GPU_DENSE_COPY);
// #undef TF_CALL_GPU_AND_ADDITIONAL_TYPES
// #undef GPU_DENSE_COPY
// #endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// #undef CPU_DENSE_COPY
// #undef INSTANTIATE_GET_VARIANT_COPY_FN

}  // namespace tensorflow
