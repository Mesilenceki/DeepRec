/* Copyright 2023 The DeepRec Authors. All Rights Reserved.
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

#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

// REGISTER_OP("FilterStorage")
//     .Input("resource: resource")
//     .Attr("partition_id: int = 0")
//     .Attr("new_partition_nums: int >= 1 = 1")
//     .Output("keys: Tkeys")
//     .Output("values: dtype")
//     .Output("versions: int64")
//     .Output("freqs: int64")
//     .Attr("Tkeys: {int64, int32}")
//     .Attr("dtype: type")
//     .Doc(R"(
// Input current parition_id embedding variable.Filter redundent ids
//     according to partition num.
// )");

// REGISTER_OP("ImportStorage")
//     .Input("resource: resource")
//     .Input("keys: partition_nums * Tkeys")
//     .Input("values: partition_nums * dtype")
//     .Input("versions: partition_nums * int64")
//     .Input("freqs: partition_nums * int64")
//     .Attr("partition_id: int = 0")
//     .Attr("partition_nums: int >= 1 = 1")
//     .Attr("Tkeys: {int64, int32}")
//     .Attr("dtype: type")
//     .Doc(R"(
// Input current parition_id embedding variable and ids from other partion
//     embedding variables.Load them according to new partition_num.
// )");

template<typename TKey, typename TValue>
class FilterStorageOp : public OpKernel {
 public:
  explicit FilterStorageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("new_partition_nums", &new_partition_nums_));
  }
  
  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue> *embedding_var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &embedding_var));
    core::ScopedUnref unref_me(embedding_var);
    std::vector<TKey> filtered_keys;
    std::vector<ValuePtr<TValue>*> value_ptr_list;
    int64 before_size = embedding_var->Size();
    OP_REQUIRES_OK(ctx, embedding_var->GetSnapshot(&filtered_keys, &value_ptr_list,
                                                   partition_id_, new_partition_nums_));

    Tensor* unneeded_ids_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {filtered_keys.size()}, &unneeded_ids_tensor));
    Tensor* unneeded_value_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {filtered_keys.size(), embedding_var->ValueLen()}, &unneeded_value_tensor));
    Tensor* version_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {filtered_keys.size()}, &version_tensor));
    Tensor* freq_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, {filtered_keys.size()}, &freq_tensor));
    if (filtered_keys.size() == 0) {
        return;
    }
    auto unneeded_ids = unneeded_ids_tensor->flat<TKey>().data();
    auto unneeded_value = unneeded_value_tensor->flat<TValue>().data();
    auto versions = version_tensor->flat<int64>().data();
    auto freq = freq_tensor->flat<int64>().data();
    embedding_var->FilterStorage(unneeded_ids, unneeded_value, versions, freq,
                                 filtered_keys, value_ptr_list);

    for (int i = 0; i < filtered_keys.size(); ++i) {
      LOG(INFO) << " key is : " << unneeded_ids[i];
      for (int j = 0; j < embedding_var->ValueLen(); ++j) {
        LOG(INFO) << " value is : " << unneeded_value[i*embedding_var->ValueLen() + j];
      }
    }
    

    int64 after_size = embedding_var->Size();
    if (before_size - after_size > 0) {
      LOG(INFO) << "JUNQI ===> filter: "<< embedding_var->Name() << " === " << before_size - after_size;
    }
  }
 private:
  int partition_id_;
  int new_partition_nums_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type) \
  REGISTER_KERNEL_BUILDER(                         \
        Name("FilterStorage")                    \
            .Device(DEVICE_CPU)                    \
            .TypeConstraint<key_type>("Tkeys")     \
            .TypeConstraint<value_type>("dtype"),  \
        FilterStorageOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

template<typename TKey, typename TValue>
class ImportStorageOp : public OpKernel {
 public:
  explicit ImportStorageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_id", &partition_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_nums", &partition_nums_));
  }
  
  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue> *embedding_var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &embedding_var));
    core::ScopedUnref unref_me(embedding_var);
    int64 before_size = embedding_var->Size();
    for (int i = 0; i < partition_nums_; ++i) {
      const Tensor& import_ids_tensor = ctx->input(1+i);
      auto* import_ids = import_ids_tensor.flat<TKey>().data();
      int64 N = import_ids_tensor.NumElements();
      if ( N == 0) continue;
      const Tensor& import_values_tensor = ctx->input(1+partition_nums_+i);
      auto* import_values = import_values_tensor.flat<float>().data();
      for (int k = 0; k < N; ++k) {
        LOG(INFO) << "key is" << import_ids[k];
        for (int j = 0; j < embedding_var->ValueLen(); ++j) {
          LOG(INFO) << "Import value is : " << import_values[k*embedding_var->ValueLen() + j];
        }
      }
      
      const Tensor& import_versions_tensor = ctx->input(1+partition_nums_*2+i);
      auto* import_versions = import_versions_tensor.flat<int64>().data();
      const Tensor& import_freqs_tensor = ctx->input(1+partition_nums_*3+i);
      auto* import_freqs = import_freqs_tensor.flat<int64>().data();    
      OP_REQUIRES_OK(ctx, embedding_var->ImportStorage(N, partition_id_, (partition_nums_+1), 
                                   import_ids, import_values, import_versions,
                                   import_freqs));
    }
    int64 after_size = embedding_var->Size();
    if (after_size - before_size > 0) {
      LOG(INFO) << "JUNQI ===> import: "<< embedding_var->Name() << " === " << after_size - before_size;
    }
    

  }
 private:
  int partition_id_;
  int partition_nums_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type) \
  REGISTER_KERNEL_BUILDER(Name("ImportStorage")    \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<key_type>("Tkeys")       \
          .TypeConstraint<value_type>("dtype"),    \
      ImportStorageOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

} // namespace tensorflow