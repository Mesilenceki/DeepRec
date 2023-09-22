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
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CKPT_DATA_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CKPT_DATA_
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/embedding_var_dump_iterator.h"
namespace tensorflow {
class BundleWriter;

namespace embedding {

template<class K, class V>
class  EmbeddingVarCkptData {
 public:
  void Emplace(K key, ValuePtr<V>* value_ptr,
               const EmbeddingConfig& emb_config,
               V* default_value, int64 value_offset,
               bool is_save_freq,
               bool is_save_version,
               bool save_unfiltered_features);

  void Emplace(K key, V* value_ptr);

  void SetWithPartition(
      std::vector<EmbeddingVarCkptData<K, V>>& ev_ckpt_data_parts);

  Status ExportToCkpt(const string& tensor_name,
                      BundleWriter* writer,
                      int64 value_len,
                      ValueIterator<V>* value_iter = nullptr) {
    size_t bytes_limit = 8 << 20;
    std::unique_ptr<char[]> dump_buffer(new char[bytes_limit]);
    LOG(INFO) << "CALLING ExportToCkpt ===> " << tensor_name
              << " ==== " << key_vec_.size() << " === " << value_ptr_vec_.size()
              << " ==== " << version_vec_.size() << " === " << freq_vec_.size();

    EVVectorDataDumpIterator<K> key_dump_iter(key_vec_);
    Status s = SaveTensorWithFixedBuffer(
        tensor_name + "-keys", writer, dump_buffer.get(),
        bytes_limit, &key_dump_iter,
        TensorShape({key_vec_.size()}));
    if (!s.ok())
      return s;
    EV2dVectorDataDumpIterator<V> value_dump_iter(
        value_ptr_vec_, value_len, value_iter);
    s = SaveTensorWithFixedBuffer(
        tensor_name + "-values", writer, dump_buffer.get(),
        bytes_limit, &value_dump_iter,
        TensorShape({value_ptr_vec_.size(), value_len}));
    if (!s.ok())
      return s;
    EVVectorDataDumpIterator<int64> version_dump_iter(version_vec_);
    s = SaveTensorWithFixedBuffer(
        tensor_name + "-versions", writer, dump_buffer.get(),
        bytes_limit, &version_dump_iter,
        TensorShape({version_vec_.size()}));
    if (!s.ok())
      return s;
    EVVectorDataDumpIterator<int64> freq_dump_iter(freq_vec_);
    s = SaveTensorWithFixedBuffer(
        tensor_name + "-freqs", writer, dump_buffer.get(),
        bytes_limit, &freq_dump_iter,
        TensorShape({freq_vec_.size()}));
    if (!s.ok())
      return s;
    LOG(INFO) << "CALLING keys_filtered";
    EVVectorDataDumpIterator<K> filtered_key_dump_iter(key_filter_vec_);
    s = SaveTensorWithFixedBuffer(
        tensor_name + "-keys_filtered", writer, dump_buffer.get(),
        bytes_limit, &filtered_key_dump_iter,
        TensorShape({key_filter_vec_.size()}));
    if (!s.ok())
      return s;
    LOG(INFO) << "CALLING versions_filtered";
    EVVectorDataDumpIterator<int64>
        filtered_version_dump_iter(version_filter_vec_);
    s = SaveTensorWithFixedBuffer(
        tensor_name + "-versions_filtered",
        writer, dump_buffer.get(),
        bytes_limit, &filtered_version_dump_iter,
        TensorShape({version_filter_vec_.size()}));
    if (!s.ok())
      return s;
    LOG(INFO) << "CALLING freqs_filtered";
    EVVectorDataDumpIterator<int64>
        filtered_freq_dump_iter(freq_filter_vec_);
    s = SaveTensorWithFixedBuffer(
        tensor_name + "-freqs_filtered",
        writer, dump_buffer.get(),
        bytes_limit, &filtered_freq_dump_iter,
        TensorShape({freq_filter_vec_.size()}));
    if (!s.ok())
      return s;
    LOG(INFO) << "CALLING partition_offset";
    EVVectorDataDumpIterator<int32>
        part_offset_dump_iter(part_offset_);
    s = SaveTensorWithFixedBuffer(
        tensor_name + "-partition_offset",
        writer, dump_buffer.get(),
        bytes_limit, &part_offset_dump_iter,
        TensorShape({part_offset_.size()}));
    if (!s.ok())
      return s;
    LOG(INFO) << "CALLING partition_filter_offset";
    EVVectorDataDumpIterator<int32>
        part_filter_offset_dump_iter(part_filter_offset_);
    s = SaveTensorWithFixedBuffer(
        tensor_name + "-partition_filter_offset",
        writer, dump_buffer.get(),
        bytes_limit, &part_filter_offset_dump_iter,
        TensorShape({part_filter_offset_.size()}));
    if (!s.ok())
      return s;

    return Status::OK();
  }

 private:
  std::vector<K> key_vec_;
  std::vector<V*> value_ptr_vec_;
  std::vector<int64> version_vec_;
  std::vector<int64> freq_vec_;
  std::vector<K> key_filter_vec_;
  std::vector<int64> version_filter_vec_;
  std::vector<int64> freq_filter_vec_;
  std::vector<int32> part_offset_;
  std::vector<int32> part_filter_offset_;
  const int kSavedPartitionNum = 1000;
};
} //namespace embedding
} //namespace tensorflow
#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CKPT_DATA_
