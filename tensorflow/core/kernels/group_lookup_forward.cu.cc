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
=======================================================================*/

#include <inttypes.h>

#include <exception>
#include <string>

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda_runtime.h>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/core/spin_rw_lock.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/stream_executor/stream_executor.h"
namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

const char* kInferenceMode = "INFERENCE_MODE";

template <typename TKey, typename TValue, typename TIndice>
struct GroupEmbeddingForWardArgs {
  TValue* emb_variable_;
  TValue* emb_vector_;
  TValue* sp_weights_;
  TIndice* unique_indices_;
  int64_t* sp_indices_;
  int* offset_indices_;
  int origin_nnz_;
};

/*
template <typename TKey, typename TValue, Combiner combiner>
__global__ void WeightedEmbeddingVarComputeFn(
    const int batch_size, const int dimension, const float max_norm,
    const int num_lookups, GroupEmbeddingForWardArgs<TKey, TValue, TIndice>* args) {
  __shared__ TValue l2_sum[1];

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < dimension) {
    for (int ev_id = 0; ev_id < num_lookups; ++ev_id) {
      int value_offset = args[ev_id].offset_indices_[bid];
      int feature_num;
      if (bid == int(batch_size) - 1) {
        feature_num = int(args[ev_id].nnz_) - value_offset;
      } else {
        feature_num = args[ev_id].offset_indices_[bid + 1] - value_offset;
      }
      TValue out = 0.0;

// #pragma unroll
      for (int j = 0; j < feature_num; ++j) {
        int64_t feature_offset = (value_offset + j) * dimension;
        TValue sum = args[ev_id].emb_variable_[feature_offset + tid];
        TValue sp_weights = args[ev_id].sp_weights_[value_offset + j];
        if (max_norm >= 0.0) {
          if (tid == 0) {
            l2_sum[0] = 0.0;
          }
          __syncthreads();
          atomicAdd(l2_sum, sum * sum);
          __syncthreads();
          TValue l2_norm = sqrtf(l2_sum[0]);
          if (l2_norm > max_norm) {
            sum *= max_norm / l2_norm;
          }
        }
        out += sum * sp_weights;
      }

      out = Combine<combiner>(out, feature_num);
      args[ev_id].emb_vector_[bid * dimension + tid] = out;
    }
  }
}*/

template <typename TKey, typename TValue, typename TIndice, Combiner combiner>
__global__ void WeightedVariableComputeFn(
    const int batch_size, const int emb_vec_size, const float max_norm,
    const int num_lookups, TValue* default_value, GroupEmbeddingForWardArgs<TKey, TValue, TIndice>* args) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  for (int ev_id = 0; ev_id < num_lookups; ++ev_id) {
    if (tid < args[ev_id].origin_nnz_) {
      int dst = args[ev_id].sp_indices_[tid];
      int* d_offset = args[ev_id].offset_indices_ + dst;
      atomicAdd(d_offset, 1);
      TIndice src = args[ev_id].unique_indices_[tid];
//      TValue weight = args[ev_id].sp_weights_[tid];
      for (int d = 0; d < emb_vec_size; ++d) {
        TValue accu = args[ev_id].emb_variable_[src + d];
        TValue* output = args[ev_id].emb_vector_ + dst + d;
        atomicAdd(output, accu);
      }

      // if (max_norm >= 0.0f) {
      //   // calc l2 norm of this emb row(per block) and compare with
      //   // max_norm.
      //   // if greater than max_norm, then clip every element with factor
      //   // max_norm / l2norm
      //   if (tid == 0) {
      //     l2_sum[0] = 0.0f;
      //   }
      //   __syncthreads();
      //   atomicAdd(l2_sum, emb_element * emb_element);
      //   __syncthreads();
      //   TValue l2_norm = sqrtf(l2_sum[0]);
      //   if (l2_norm > max_norm) {
      //     emb_element *= max_norm / l2_norm;
      //   }
      // }
    }
    __syncthreads();
    if (tid < batch_size) {
      int feature_num = args[ev_id].offset_indices_[tid];
      if (feature_num > 0) {
        for (int d = 0; d < emb_vec_size; ++d) {
          TValue out = args[ev_id].emb_vector_[tid * emb_vec_size + d];
	        args[ev_id].emb_vector_[tid * emb_vec_size + d] = Combine<combiner>(out, feature_num);
        }
      } else {
        for (int d = 0; d < emb_vec_size; ++d) {
          args[ev_id].emb_vector_[tid * emb_vec_size + d] = default_value[d];
        }
      }
    }
  }
}

template <typename TKey, typename TValue, typename TIndice, Combiner combiner>
__global__ void EmbeddingVarComputeFn(
    const int batch_size, const int dimension, const float max_norm,
    const int num_lookups, TValue* default_value, GroupEmbeddingForWardArgs<TKey, TValue, TIndice>* args) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  for (int ev_id = 0; ev_id < num_lookups; ++ev_id) {
    if (tid < args[ev_id].origin_nnz_) {
      int dst = args[ev_id].sp_indices_[tid];
      int* d_offset = args[ev_id].offset_indices_ + dst;
      atomicAdd(d_offset, 1);
      TIndice src = args[ev_id].unique_indices_[tid];
//      TValue weight = args[ev_id].sp_weights_[tid];
      //printf("dst addrs is %d src addrs is %d\n", dst, src);
      for (int d = 0; d < dimension; ++d) {
        TValue accu = args[ev_id].emb_variable_[src + d];
        TValue* output = args[ev_id].emb_vector_ + dst + d;
        atomicAdd(output, accu);
      }

      // if (max_norm >= 0.0f) {
      //   // calc l2 norm of this emb row(per block) and compare with
      //   // max_norm.
      //   // if greater than max_norm, then clip every element with factor
      //   // max_norm / l2norm
      //   if (tid == 0) { 
      //     l2_sum[0] = 0.0f;
      //   }
      //   __syncthreads();
      //   atomicAdd(l2_sum, emb_element * emb_element);
      //   __syncthreads();
      //   TValue l2_norm = sqrtf(l2_sum[0]);
      //   if (l2_norm > max_norm) {
      //     emb_element *= max_norm / l2_norm;
      //   }
      // }
    }
    __syncthreads();
    if (tid < batch_size) {
      int feature_num = args[ev_id].offset_indices_[tid];
      if (feature_num > 0) {
        for (int d = 0; d < dimension; ++d) {
          TValue out = args[ev_id].emb_vector_[tid * dimension + d];
	  args[ev_id].emb_vector_[tid * dimension + d] = Combine<combiner>(out, feature_num);
        }
      } else {
        for (int d = 0; d < dimension; ++d) {
          args[ev_id].emb_vector_[tid * dimension + d] = default_value[d];
        }
      }
    }
  }
}

template <typename TKey, typename TValue>
__global__ void VariableComputeFn(
    const int unique_nnz, const int emb_vec_size, const int64_t emb_dim_limit,
    const TKey* keys, const TValue* embeddings, TValue* outputs) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < unique_nnz) {
    int indices = keys[tid];
    TValue emb_out = 0.0;
    if (FastBoundsCheck(indices, emb_dim_limit)) {
#pragma unroll
      for (int d = 0; d < emb_vec_size; ++d) {
        emb_out = embeddings[indices * emb_vec_size + d];
        outputs[tid * emb_vec_size + d] = emb_out;
      }   
    }
  }
}

}  // namespace

template <typename TKey, typename TValue, typename TIndice>
class GroupEmbeddingLookupForWard {
 public:
  void initialize(const int num_lookups, const int dimension,
                  const float max_norm) {
    max_norm_ = max_norm;
    dimension_ = dimension;
    ev_nums_ = num_lookups;
    args_size_ = sizeof(GroupEmbeddingForWardArgs<TKey, TValue, TIndice>);
    CK_CUDA_THROW_(cudaMalloc(&d_args_, args_size_ * num_lookups));
    h_args_.resize(ev_nums_);
  }

  ~GroupEmbeddingLookupForWard() {
    if (d_args_) {
      CK_CUDA_THROW_(cudaFree(d_args_));
    }
  }

  void set(int idx, TValue* emb_variable, TValue* emb_vector,
           int* offset_indices, int origin_nnz, TValue* sp_weights,
           TIndice* unique_indices, int64_t* sp_indices) {
    h_args_[idx].emb_variable_ = emb_variable;
    h_args_[idx].emb_vector_ = emb_vector;
    h_args_[idx].offset_indices_ = offset_indices;
    h_args_[idx].sp_indices_ = sp_indices;
    h_args_[idx].origin_nnz_ = origin_nnz;
    h_args_[idx].sp_weights_ = sp_weights;
    h_args_[idx].unique_indices_ = unique_indices;
  }

  template <typename ForwardFn>
  void compute(ForwardFn compute_fn, TValue* default_value, const int batch_size,
               cudaStream_t stream) {
    CK_CUDA_THROW_(cudaMemcpyAsync(d_args_, h_args_.data(),
                                   ev_nums_ * args_size_,
                                   cudaMemcpyHostToDevice, stream));

    CK_CUDA_THROW_(cudaStreamSynchronize(stream));

    {
      // TODO: double check why mapped 2D grid slower
      const int threads = 1024ul;
      dim3 grid_size{65536/threads, 1};
      compute_fn<<<grid_size, threads, 0, stream>>>(
          batch_size, dimension_, max_norm_, ev_nums_, default_value, d_args_);
    }

    CK_CUDA_THROW_(cudaGetLastError());
  }

 protected:
  std::vector<GroupEmbeddingForWardArgs<TKey, TValue, TIndice>> h_args_;
  GroupEmbeddingForWardArgs<TKey, TValue, TIndice>* d_args_{nullptr};
  float max_norm_{0.0f};
  int ev_nums_{0};
  int dimension_{0};
  size_t args_size_{0};
};

template <typename TKey, typename TValue, typename TIndice>
class GroupEmbeddingLookupForwardBaseOp : public OpKernel {
 public:
  explicit GroupEmbeddingLookupForwardBaseOp(OpKernelConstruction *c)
      : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    OP_REQUIRES_OK(c, c->GetAttr("ignore_weights", &ignore_weights_));
    lookuper_.initialize(num_lookups_, dimension_, max_norm_);
  }

  inline void Lookup(const bool is_ev, const int batch_size, cudaStream_t stream) { 
    if (ignore_weights_) {
    //   if (combiner_ == "mean") {
    //     if (is_ev) {
    //       lookuper_.compute(EmbeddingVarComputeFn<TKey, TValue, Mean>, batch_size, stream);
    //     } else {
    //       lookuper_.compute(VariableComputeFn<TKey, TValue, Mean>, batch_size, stream);
    //     }
    //   } else if (this->combiner_ == "sum") {
    //     if (is_ev) {
    //       lookuper_.compute(EmbeddingVarComputeFn<TKey, TValue, Sum>, batch_size, stream);
    //     } else {
    //       lookuper_.compute(VariableComputeFn<TKey, TValue, Sum>, batch_size, stream);
    //     }
    //   } else {
    //     if (is_ev) {
    //       lookuper_.compute(EmbeddingVarComputeFn<TKey, TValue, Sqrtn>, batch_size, stream);
    //     } else {
    //       lookuper_.compute(VariableComputeFn<TKey, TValue, Sqrtn>, batch_size, stream);
    //     }
    //   }
    // } else {
    //   if (combiner_ == "mean") {
    //     if (is_ev) {
    //       lookuper_.compute(WeightedEmbeddingVarComputeFn<TKey, TValue, Mean>, batch_size, stream);
    //     } else {
    //       lookuper_.compute(WeightedVariableComputeFn<TKey, TValue, Mean>, batch_size, stream);
    //     }
    //   } else if (this->combiner_ == "sum") {
    //     if (is_ev) {
    //       lookuper_.compute(WeightedEmbeddingVarComputeFn<TKey, TValue, Sum>, batch_size, stream);
    //     } else {
    //       lookuper_.compute(WeightedVariableComputeFn<TKey, TValue, Sum>, batch_size, stream);
    //     }
    //   } else {
    //     if (is_ev) {
    //       lookuper_.compute(WeightedEmbeddingVarComputeFn<TKey, TValue, Sqrtn>, batch_size, stream);
    //     } else {
    //       lookuper_.compute(WeightedVariableComputeFn<TKey, TValue, Sqrtn>, batch_size, stream);
    //     }
    //   }
    }
  }

 protected:
  GroupEmbeddingLookupForWard<TKey, TValue, TIndice> lookuper_;
  std::string combiner_;
  float max_norm_;
  int num_lookups_;
  int dimension_;
  bool ignore_weights_;
};

template <typename TFKey, typename TKey, typename TValue, typename TIndice>
class GroupEmbeddingVarLookupOp
    : public GroupEmbeddingLookupForwardBaseOp<TKey, TValue, TIndice> {
 public:
  explicit GroupEmbeddingVarLookupOp(OpKernelConstruction* c)
      : GroupEmbeddingLookupForwardBaseOp<TKey, TValue, TIndice>(c) {
    bool is_inference;
    TF_CHECK_OK(ReadBoolFromEnvVar(kInferenceMode, false, &is_inference));
    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue *default_v, TFKey id, int64 index,
                             int64 total_dim,
                             int64 len) { return default_v + len * index; };
    } else {
      get_default_v_fn_ = [](TValue *default_v, TFKey id, int64 index,
                             int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim);
      };
    }
    if (!is_inference) {
      lookup_fn_ = [](EmbeddingVar<TFKey, TValue>* ev, const TFKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device) {
        ev->LookupOrCreate(key, val, default_v, default_v_num,
            is_use_default_value_tensor, n, device);
      };
    } else {
      lookup_fn_ = [](EmbeddingVar<TFKey, TValue>* ev, const TFKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device) {
        ev->Lookup(key, val, default_v, default_v_num,
            is_use_default_value_tensor, n, device);
      };
    }
    tensor_list_.reserve(this->num_lookups_);
  }

  ~GroupEmbeddingVarLookupOp() { delete[] occupy_flag_; }

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TFKey, TValue>* ev = nullptr;
    const auto& device = ctx->eigen_device<GPUDevice>();
    int64_t batch_size = -1;
    TValue *default_v = nullptr;

    for (size_t i = 0; i < this->num_lookups_; ++i) {
      const Tensor& unique_values_tensor = ctx->input(this->num_lookups_ + i);
      auto unique_values = unique_values_tensor.flat<TFKey>();
      int64 N = unique_values_tensor.NumElements();

      const Tensor& unique_indices_tensor = ctx->input(this->num_lookups_ * 2 + i);
      const Tensor& sp_indices_tensor = ctx->input(this->num_lookups_ * 3 + i);
      auto sp_indices = sp_indices_tensor.flat<int64>().data();
      int origin_nnz = sp_indices_tensor.shape().dim_size(0);

      if (i == 0) {
        const Tensor& dense_shape_tensor =
            ctx->input(this->num_lookups_ * 4 + i);
        auto dense_shape = dense_shape_tensor.flat<int64>().data();
        batch_size = dense_shape[0];
      }

      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, i), &ev));
      core::ScopedUnref unref_me(ev);
      if (is_use_default_value_tensor_) {
        default_v = (TValue *)ctx->input(6 * this->num_lookups_).data();
      } else {
        default_v = ev->GetDefaultValuePtr();
      }
      // DEBUG
      int64 dimension = ev->ValueLen();
      // DEBUG
      const TFKey* key_base = unique_values.data();
      Tensor out_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TValue>::value,
                                             {N * dimension}, &out_tensor));
      TValue* out_base = out_tensor.flat<TValue>().data();

      if (ev->IsSingleHbm()) {
        if (is_use_default_value_tensor_) {
          Tensor default_values(ctx->input(5 * this->num_lookups_));
          auto default_value_num = default_values.NumElements() / dimension;
          auto default_values_matrix =
              default_values.shaped<TValue, 2>({default_value_num, dimension});
          TValue* default_v_base = &default_values_matrix(0, 0);
          lookup_fn_(ev, key_base, out_base, default_v_base,
                      default_value_num, is_use_default_value_tensor_, N,
                      device);
        } else {
          lookup_fn_(ev, key_base, out_base, ev->GetDefaultValuePtr(),
                      ev->GetDefaultValueDim(), true, N, device);
        }
      } else {
        auto out_flat =
            out_tensor.shaped<TValue, 2>({N, out_tensor.NumElements() / N});
        const int64 slice_elems = out_flat.dimension(1);
        const size_t slice_bytes = slice_elems * sizeof(TValue);
        TValue** memcpy_address = new TValue* [N];
        TFKey* indices_host = new TFKey[N];

        auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
        int64 num_threads = worker_threads->num_threads;
        if (occupy_flag_ == nullptr) {
          mutex_lock l(m_init_occupy_flag_);
          // double check
          if (occupy_flag_ == nullptr) {
            occupy_flag_ = new bool[num_threads];
            memset(occupy_flag_, 0, sizeof(bool) * num_threads);
          }
        }
        std::vector<std::list<int64>> init_cursor_list(num_threads + 1);
        std::vector<std::list<int64>> copyback_cursor_list(num_threads + 1);

        volatile bool is_cpu_indices_ready = false;
        // Copy ids from GPU to CPU for CPU Lookup.
        auto stream = ctx->op_device_context()->stream();
        auto event_mgr = ctx->device()->tensorflow_gpu_device_info()->event_mgr;

        se::DeviceMemoryBase gpu_src(const_cast<TFKey *>(key_base),
                                     N * sizeof(TFKey));
        stream->ThenMemcpy(indices_host, gpu_src, N * sizeof(TFKey));
        SyncWithEventMgr(stream, event_mgr);

        uint64 main_thread_id = Env::Default()->GetCurrentThreadId();
        auto do_work = [this, indices_host, out_base, slice_elems, ctx, ev,
                        memcpy_address, &init_cursor_list,
                        &copyback_cursor_list, main_thread_id,
                        num_threads](int64 start, int64 limit) {
          uint64 thread_id = Env::Default()->GetCurrentThreadId();
          int64 position;
          if (thread_id == main_thread_id) {
            position = num_threads;
          } else {
            position = -1;
            {
              spin_rd_lock l(mu_);
              auto iter = hash_map_.find(thread_id);
              if (iter != hash_map_.end()) {
                position = iter->second;
              }
            }

            if (position == -1) {
              // bind a new thread to a local cursor_list
              position = thread_id % num_threads;
              while (!__sync_bool_compare_and_swap(&(occupy_flag_[position]),
                                                   false, true)) {
                position = (position + 1) % num_threads;
              }
              {
                spin_wr_lock l(mu_);
                hash_map_.insert(std::pair<uint64, int64>(thread_id, position));
              }
            }
          }
          ev->LookupWithFreqBatch(indices_host, memcpy_address, start, limit,
                                  init_cursor_list[position],
                                  copyback_cursor_list[position]);
        };
        Shard(num_threads, worker_threads->workers, N, slice_bytes, do_work);
        for (int i = 1; i < num_threads + 1; i++) {
          if (init_cursor_list[i].size() > 0) {
            init_cursor_list[0].splice(init_cursor_list[0].end(),
                                       init_cursor_list[i]);
          }
          if (copyback_cursor_list[i].size() > 0) {
            copyback_cursor_list[0].splice(copyback_cursor_list[0].end(),
                                           copyback_cursor_list[i]);
          }
        }
        // Pointers in memcpy_address here will
        // be cast to ValuePtr<Tvalue>* in this funcation.
        ev->AllocateMemoryForNewFeatures(memcpy_address, init_cursor_list[0]);

        ev->SetDefaultValueOfNewFeatures(
            indices_host, N, init_cursor_list[0], memcpy_address, default_v,
            get_default_v_fn_, stream, event_mgr, ctx->eigen_gpu_device());

        ev->CopyEmbeddingsFromCPUToGPU(indices_host, copyback_cursor_list[0],
                                       memcpy_address, stream, event_mgr,
                                       ctx->eigen_gpu_device(), worker_threads);

        ev->CopyEmbeddingsToBuffer(out_base, N, slice_elems, memcpy_address,
                                   stream, event_mgr, ctx->eigen_gpu_device());
        delete[] memcpy_address;

        if (ev->IsMultiLevel()) {
          ev->storage_manager()->Schedule([ev, indices_host, N]() {
            embedding::BatchCache<TFKey> *cache = ev->Cache();
            cache->add_to_rank(indices_host, N);
            delete[] indices_host;
          });
        }
      }

      Tensor *op_output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {batch_size, dimension},
                                               &op_output_tensor));
      auto op_output = op_output_tensor->flat<TValue>().data();

      // allocate offset tensor
      TensorShape values_offset_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(batch_size)}));

      Tensor* values_offset_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ + i,
                                               values_offset_tensor_shape,
                                               &values_offset_tensor));
      auto values_offset = values_offset_tensor->flat<int>().data();
      
      TValue* sp_weights = nullptr;
      if (!this->ignore_weights_) {
        const Tensor &sp_weights_tensor = ctx->input(this->num_lookups_ * 5 + i);
        sp_weights = const_cast<TValue*>(sp_weights_tensor.flat<TValue>().data());
      }

      // auto d_indices = unique_indices_tensor.flat<TIndice>().data();
      // TIndice* h_value = new TIndice[unique_indices_tensor.NumElements()];
      // cudaMemcpy(h_value, d_indices, sizeof(TIndice) * unique_indices_tensor.NumElements(), 
      //     cudaMemcpyDeviceToHost);
      // for (int t = 0 ; t < unique_indices_tensor.NumElements(); ++t) {
      //   std::cout << h_value[t] << "++\n";
      // }
      // std::cout << " ================= " << std::endl;
      // auto d_sp_indices = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(
      //         sp_indices_tensor.flat<int64>().data()));
      // TValue* h_sp_value = new TValue[N * dimension];
      // cudaMemcpy(h_sp_value, out_base, sizeof(TValue) * N * dimension, 
      //     cudaMemcpyDeviceToHost);
      // for (int t = 0 ; t < N * dimension; ++t) {
      //   std::cout << h_sp_value[t] << "=====\n";
      // }

      this->lookuper_.set(
          i, out_base, op_output, values_offset, origin_nnz, sp_weights,
          const_cast<TIndice*>(unique_indices_tensor.flat<TIndice>().data()),
          const_cast<int64_t *>(reinterpret_cast<const int64_t *>(sp_indices)));

      tensor_list_.emplace_back(out_tensor);
    }
    
    if (this->combiner_ == "sum") {
      this->lookuper_.compute(EmbeddingVarComputeFn<TKey, TValue, TIndice, Sum>, default_v, batch_size, device.stream());
    } else if (this->combiner_ == "mean") {
      this->lookuper_.compute(EmbeddingVarComputeFn<TKey, TValue, TIndice, Mean>, default_v, batch_size, device.stream());
    } else {
      this->lookuper_.compute(EmbeddingVarComputeFn<TKey, TValue, TIndice, Sqrtn>, default_v, batch_size, device.stream());
    }

    tensor_list_.clear();
  }

 private:
  std::vector<Tensor> tensor_list_;
  std::map<uint64, int64> hash_map_;
  std::function<TValue *(TValue *, TFKey, int64, int64, int64)>
      get_default_v_fn_;
  std::function<void(EmbeddingVar<TFKey, TValue>* ev, const TFKey* key,
                      TValue* val, TValue* default_v, int32 default_v_num,
                      bool is_use_default_value_tensor,
                      size_t n, const Eigen::GpuDevice& device)> lookup_fn_;
  mutable easy_spinrwlock_t mu_ = EASY_SPINRWLOCK_INITIALIZER;
  bool* occupy_flag_{nullptr};
  mutex m_init_occupy_flag_;
  bool is_use_default_value_tensor_;
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype, indice_type) \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("GroupEmbeddingVarLookup")                                \
          .Device(DEVICE_GPU)                                        \
          .HostMemory("dense_shape")                                 \ 
          .TypeConstraint<key_type_tf>("Tkeys")                      \
          .TypeConstraint<dtype>("dtype")                            \
          .TypeConstraint<indice_type>("TIndices"),                   \
      GroupEmbeddingVarLookupOp<key_type_tf, key_type, dtype, indice_type>)

REGISTER_GPU_KERNELS(int64, int64_t, float, float, int32);
REGISTER_GPU_KERNELS(int32, int32_t, float, float, int32);
REGISTER_GPU_KERNELS(int64, int64_t, float, float, int64);
REGISTER_GPU_KERNELS(int32, int32_t, float, float, int64);
#undef REGISTER_GPU_KERNELS

template <typename TFKey, typename TKey, typename TValue, typename TIndice>
class GroupVariableLookupOp
    : public GroupEmbeddingLookupForwardBaseOp<TKey, TValue, TIndice> {
 public:
  explicit GroupVariableLookupOp(OpKernelConstruction* c)
      : GroupEmbeddingLookupForwardBaseOp<TKey, TValue, TIndice>(c) {
    tensor_list_.reserve(this->num_lookups_);
  }

  void Compute(OpKernelContext* ctx) override {
    auto device = ctx->eigen_device<GPUDevice>();
    const cudaStream_t stream  = device.stream();
    int batch_size = -1;
    const Tensor& default_value_tensor = ctx->input(this->num_lookups_ * 6);
    TValue* default_value = const_cast<TValue*>(default_value_tensor.flat<TValue>().data());

    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor& emb_variable_tensor = ctx->input(i);
      const Tensor& unique_values_tensor = ctx->input(this->num_lookups_ + i);
      auto unique_values = unique_values_tensor.flat<TFKey>().data();
      int64 emb_row_size = emb_variable_tensor.shape().dim_size(0);
      int64 emb_vec_size = emb_variable_tensor.shape().dim_size(1);

      const Tensor& unique_indices_tensor = ctx->input(this->num_lookups_ * 2 + i);
      int unique_nnz = unique_values_tensor.NumElements();
      int origin_nnz = unique_indices_tensor.NumElements();

      const Tensor& sp_indices_tensor = ctx->input(this->num_lookups_ * 3 + i);

      if (i == 0) {
        const Tensor& dense_shape_tensor =
            ctx->input(this->num_lookups_ * 4 + i);
        auto dense_shape = dense_shape_tensor.flat<int64>().data();
        batch_size = dense_shape[0];
      }

      const auto* key_base = const_cast<TKey *>(reinterpret_cast<const TKey *>(unique_values));
      Tensor out_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TValue>::value,
                                             {unique_nnz * emb_vec_size}, &out_tensor));
      TValue* out_base = out_tensor.flat<TValue>().data();
      tensor_list_.emplace_back(out_tensor);

      dim3 grid_size{65536 / 32, 1};
      dim3 block_size{32, 1};

      VariableComputeFn<<<grid_size, block_size, 0 , stream>>>(
          unique_nnz, emb_vec_size, emb_row_size, key_base ,
          reinterpret_cast<const TValue *>(
            emb_variable_tensor.flat<TValue>().data()), out_base);

      TensorShape emb_vectors_tensor_shape = TensorShape(std::vector<int64>(
          {static_cast<long long>(batch_size), emb_vec_size}));

      Tensor* emb_vectors_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &emb_vectors_tensor));
      auto emb_vectors = emb_vectors_tensor->flat<TValue>().data();

      // allocate offset tensor
      TensorShape values_offset_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(batch_size)}));
      Tensor* values_offset_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ + i,
                                               values_offset_tensor_shape,
                                               &values_offset_tensor));
      auto values_offset = values_offset_tensor->flat<int>().data();

      TValue* sp_weights = nullptr;
      if (!this->ignore_weights_) {
        const Tensor &sp_weights_tensor = ctx->input(this->num_lookups_ * 5 + i);
        sp_weights = const_cast<TValue*>(sp_weights_tensor.flat<TValue>().data());
      }
      
      this->lookuper_.set(
          i, out_base, emb_vectors, values_offset, origin_nnz, sp_weights,
              const_cast<TIndice*>(unique_indices_tensor.flat<TIndice>().data()),
          const_cast<int64_t *>(reinterpret_cast<const int64_t *>(
              sp_indices_tensor.flat<int64>().data())));
    }

    if (this->combiner_ == "sum") {
      this->lookuper_.compute(WeightedVariableComputeFn<TKey, TValue, TIndice, Sum>, default_value, batch_size, stream);
    } else if (this->combiner_ == "mean") {
      this->lookuper_.compute(WeightedVariableComputeFn<TKey, TValue, TIndice, Mean>, default_value, batch_size, stream);
    } else {
      this->lookuper_.compute(WeightedVariableComputeFn<TKey, TValue, TIndice, Sqrtn>, default_value, batch_size, stream);
    }
    tensor_list_.clear();
    // this->Lookup(false, batch_size, stream);
  }
 private:
  std::vector<Tensor> tensor_list_;
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype, indice_type) \
  REGISTER_KERNEL_BUILDER(Name("GroupVariableLookup")                \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("dense_shape")             \ 
                              .TypeConstraint<key_type_tf>("Tkeys")  \
                              .TypeConstraint<dtype>("dtype")       \
                              .TypeConstraint<indice_type>("TIndices"), \
                          GroupVariableLookupOp<key_type_tf, key_type, dtype, indice_type>)

REGISTER_GPU_KERNELS(int64, int64_t, float, float, int32);
REGISTER_GPU_KERNELS(int32, int32_t, float, float, int32);
REGISTER_GPU_KERNELS(int64, int64_t, float, float, int64);
REGISTER_GPU_KERNELS(int32, int32_t, float, float, int64);
#undef REGISTER_GPU_KERNELS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
