/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <bitset>
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/iterator/constant_input_iterator.cuh"
#include "cub/iterator/transform_input_iterator.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/counting_input_iterator.cuh"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/core/kernels/cuda_solvers.h"

#define TF_CHECK_CUDA_CALL(x, error_msg)                                       \
  do {                                                                         \
    cudaError_t retval = (x);                                                  \
    if (retval != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("Runtime error: ") +                \
                               (cudaGetErrorString(retval)) + " " + __FILE__ + \
                               ":" + std::to_string(__LINE__) + ":" +          \
                               std::string(error_msg) + " \n");                \
    }                                                                          \
  } while (0)

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

// Returns true iff index is at the end of a segment (which is equivalent to the
// beginning of the next segment).
template <typename TKey, typename TIndex>
struct SegmentIndicatorFunctor {
  const TKey *__restrict__ sorted_input_ptr_;
  SegmentIndicatorFunctor(const TKey *sorted_input_ptr) : sorted_input_ptr_(sorted_input_ptr) {}
  __device__ bool operator()(const TIndex &i) const {
    return i > 0 && sorted_input_ptr_[i] != sorted_input_ptr_[i - 1];
  }
};

template <typename TIndex>
__global__ void RangeInitKernel(const TIndex start, const TIndex delta,
                                const int64 size, TIndex* out) {  
  GPU_1D_KERNEL_LOOP(i, size) { out[i] = start + i * delta; }
}
template <typename TIndex>
__global__ void MoveValuesKernel(const TIndex* keys, const TIndex* values, const int64* offset_values,
                                 const int64 size, const int num_lookups, TIndex** out) {
  GPU_1D_KERNEL_LOOP(i, size) {
    TIndex key = ldg(keys + i);
    // for (int k = 0; i < num_lookups; ++k) {
    //   printf(" ========= key is %lld offset is %lld k is %d \n", key, offset_values[k], k);
    //   if (key < offset_values[k+1]) {
	  //     TIndex indice = key - offset_values[k];
    //     out[k][indice] = ldg(values + i);
    //     break;
    //   }
    // }
    out[key] = ldg(values + i);
  }
}
template <typename TIndex>
__global__ void MoveValuesKernel(const TIndex* keys, const TIndex* values,
                                 const int64* size_ptr, TIndex* out) {
  int64 size = ldg(size_ptr);
  GPU_1D_KERNEL_LOOP(i, size) {
    TIndex key = ldg(keys + i);
    out[key] = ldg(values + i);
  }
}
template <typename T, typename TIndex>
__global__ void MoveSparseValuesKernel(const TIndex* keys, const TIndex* idxs,
                                       const T* values, const int64 size,
                                       T* out) {
  GPU_1D_KERNEL_LOOP(i, size) {
    TIndex key = ldg(keys + i);
    TIndex idx = ldg(idxs + i);
    out[key] = ldg(values + idx);
  }
}
template <typename TIndex>
void RangeInit(const GPUDevice& d, const TIndex start, const TIndex delta,
               const int64 size, TIndex* out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  RangeInitKernel<TIndex>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          start, delta, size, out);  
}
template <typename TIndex>
void MoveValues(const GPUDevice& d, const TIndex* keys, const TIndex* values, const int64* offset_indices,
                const int64 size, const int num_lookups, TIndex** out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  MoveValuesKernel<<<config.block_count, config.thread_per_block, 0,
                     d.stream()>>>(keys, values, offset_indices, size, num_lookups, out);
}

template <typename TIndex>
void MoveValues(const GPUDevice& d, const TIndex* keys, const TIndex* values,
                const int64* size_ptr, const int64 size, TIndex* out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  MoveValuesKernel<<<config.block_count, config.thread_per_block, 0,
                     d.stream()>>>(keys, values, size_ptr, out);
}
template <typename T, typename TIndex>
void MoveSparseValues(const GPUDevice& d, const TIndex* keys, const TIndex* idxs,
                      const T* values, const int64 size, T* out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  MoveSparseValuesKernel<<<config.block_count, config.thread_per_block, 0,
                           d.stream()>>>(keys, idxs, values, size, out);
}
template <typename T, typename TIndex>
class GroupUniqueAliV2GpuOp : public AsyncOpKernel {
 public:
  explicit GroupUniqueAliV2GpuOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_lookups", &num_lookups_));
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    std::vector<int64> h_batch_offset{0};
    std::vector<T*> keys_vec;
    h_batch_offset.reserve(num_lookups_+1);
    keys_vec.reserve(num_lookups_);

    int64 total_N = 0;
    for (int i = 0; i < num_lookups_; ++i) {
      const Tensor& input_tensor = ctx->input(i);
      const T* keys = input_tensor.flat<T>().data();
      int64 N = input_tensor.NumElements();
      total_N += N;
      h_batch_offset.push_back(total_N);
      keys_vec.push_back(const_cast<T*>(keys));
    }
    T* d_keys = nullptr;
    cudaMalloc(&d_keys, total_N * sizeof(int64));

    const GPUDevice& device = ctx->eigen_device<GPUDevice>();
    const cudaStream_t& cu_stream = GetGpuStream(ctx);
    for (int i = 0; i < num_lookups_; ++i) {
      int num = h_batch_offset[i+1] - h_batch_offset[i];
      TF_CHECK_CUDA_CALL(cudaMemcpyAsync(d_keys + h_batch_offset[i], keys_vec[i], num * sizeof(int64),
          cudaMemcpyDeviceToDevice, cu_stream), "memcpy");
    }
    TF_CHECK_CUDA_CALL(cudaStreamSynchronize(cu_stream), "sync");

    Tensor* output_tensor = nullptr;
    Tensor* idx_tensor = nullptr;
    auto allocate_output = [ctx, &output_tensor, &idx_tensor_vec, &h_batch_offset, total_N,
                            &device, this](int64 N_out) {
      for (int i = 0; i < num_lookups_; ++i) {
        TF_RETURN_IF_ERROR(ctx->allocate_output(i, {N_out}, &output_tensor));
      }
      TF_RETURN_IF_ERROR(ctx->allocate_output(num_lookups_, {total_N}, &idx_tensor));
      return Status::OK();
    };

    if (total_N == 0) {
      OP_REQUIRES_OK_ASYNC(ctx, allocate_output(0), done);
      done();
      return;
    }
    
    Tensor keys_sort_tensor;
    Tensor indicies_sort_tensor;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, {total_N},
                                           &keys_sort_tensor), done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_temp(DataTypeToEnum<TIndex>::value, {total_N},
                                           &indicies_sort_tensor), done);
    T* keys_sort = keys_sort_tensor.flat<T>().data();
    TIndex* indicies_sort = indicies_sort_tensor.flat<TIndex>().data();
    
    Tensor indices_in_tensor;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_temp(DataTypeToEnum<TIndex>::value, {total_N},
                                           &indices_in_tensor), done);
    TIndex* indices_in = indices_in_tensor.flat<TIndex>().data();
    RangeInit(device, (TIndex)0, (TIndex)1, total_N, indices_in);

    {
      const T* keys_in;
      Tensor keys_in_tensor;
      keys_in = d_keys;
      using U = typename std::make_unsigned<T>::type;
      const U* keys_u_in = reinterpret_cast<const U*>(keys_in);
      U* keys_u_sort = reinterpret_cast<U*>(keys_sort);
      
      Tensor cub_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, keys_u_in,
                                      keys_u_sort, indices_in, indicies_sort, total_N,
                                      0, sizeof(T) * 8, cu_stream);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(
              DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
              &cub_temp_storage), done);
      cub::DeviceRadixSort::SortPairs(cub_temp_storage.flat<int8>().data(),
                                      temp_storage_bytes, keys_u_in,
                                      keys_u_sort, indices_in, indicies_sort, total_N,
                                      0, sizeof(T) * 8, cu_stream);
    }

    Tensor output_indices_tensor;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_temp(DataTypeToEnum<TIndex>::value, {total_N},
                                           &output_indices_tensor), done);
    TIndex* output_indices = output_indices_tensor.flat<TIndex>().data();

    {
      cub::TransformInputIterator<TIndex, SegmentIndicatorFunctor<T, TIndex>,
                                cub::CountingInputIterator<TIndex>>
        segment_indicator_iter(0, {keys_sort});
      Tensor cub_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, segment_indicator_iter,
                                    output_indices, total_N, cu_stream);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(
              DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
              &cub_temp_storage), done);
      cub::DeviceScan::InclusiveSum(cub_temp_storage.flat<int8>().data(),
                                    temp_storage_bytes, segment_indicator_iter,
                                    output_indices, total_N, cu_stream);
    }

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES_ASYNC(ctx, stream, errors::Internal("No GPU stream available."), done);
    ScratchSpace<TIndex> N_out(ctx, 1, /*on_host=*/true);
    se::DeviceMemoryBase wrapped_num_out(output_indices + (total_N - 1),
                                         sizeof(TIndex));
    TensorReference ref_output_indices(output_indices_tensor);
    OP_REQUIRES_ASYNC(ctx,
                stream->ThenMemcpy(N_out.mutable_data(), wrapped_num_out, sizeof(TIndex)).ok(),
                errors::Internal("Failed to launch copy from device to host."), done);
    ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, [ref_output_indices]() { ref_output_indices.Unref(); });
    stream->BlockHostUntilDone();
    int64_t uniq_size = (*N_out.data()) + 1;
    std::cout << "unique_size is" << uniq_size << std::endl;
    OP_REQUIRES_OK_ASYNC(ctx, allocate_output(uniq_size), done);
    T* output = output_tensor->flat<T>().data();
    std::vector<TIndex*> h_idx;
    h_idx.reserve(idx_tensor_vec.size());
    for (int i = 0; i < idx_tensor_vec.size(); ++i) {
      h_idx.push_back(idx_tensor_vec[i]->flat<TIndex>().data());
    }
    TIndex** d_idx;
    cudaMalloc(&d_idx, idx_tensor_vec.size() * sizeof(TIndex*));
    cudaMemcpyAsync(d_idx, h_idx.data(), idx_tensor_vec.size() * sizeof(TIndex*),
        cudaMemcpyHostToDevice, cu_stream);
    int64* d_batch_offset = nullptr;
    cudaMalloc(&d_batch_offset, (num_lookups_+1) * sizeof(int64));
    cudaMemcpyAsync(d_batch_offset, h_batch_offset.data(), (num_lookups_+1) * sizeof(int64),
        cudaMemcpyHostToDevice, cu_stream);
    TF_CHECK_CUDA_CALL(cudaStreamSynchronize(cu_stream), "sync");
    MoveValues(device, indicies_sort, output_indices, d_batch_offset, total_N, num_lookups_, d_idx);
    MoveSparseValues(device, output_indices, indicies_sort, d_keys, total_N, output);
    done();
  }
 private:
  int num_lookups_{0};
};

#define REGISTER_UNIQUE_ALI_V2_GPU_KERNEL(T, TIndex)			\
  REGISTER_KERNEL_BUILDER(Name("GroupUnique")				\
                          .Device(DEVICE_GPU)				\
                          .TypeConstraint<T>("T")			\
                          .TypeConstraint<TIndex>("out_idx"),		\
                          GroupUniqueAliV2GpuOp<T, TIndex>)
#define REGISTER_UNIQUE_ALI_V2_GPU(T)		\
  REGISTER_UNIQUE_ALI_V2_GPU_KERNEL(T, int32);	\
  REGISTER_UNIQUE_ALI_V2_GPU_KERNEL(T, int64)
  
TF_CALL_int32(REGISTER_UNIQUE_ALI_V2_GPU);
TF_CALL_int64(REGISTER_UNIQUE_ALI_V2_GPU);
TF_CALL_uint32(REGISTER_UNIQUE_ALI_V2_GPU);
TF_CALL_uint64(REGISTER_UNIQUE_ALI_V2_GPU);

#undef REGISTER_UNIQUE_ALI_V2_GPU
#undef REGISTER_UNIQUE_ALI_V2_GPU_KERNEL

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
