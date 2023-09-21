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

#include "gtest/gtest.h"

#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"


namespace tensorflow {

namespace {
  struct ProcMemory {
    long size;      // total program size
    long resident;  // resident set size
    long share;     // shared pages
    long trs;       // text (code)
    long lrs;       // library
    long drs;       // data/stack
    long dt;        // dirty pages
  };

  static long read_proc_memory() {
    ProcMemory m;
    errno = 0;
    FILE* fp = NULL;
    fp = fopen("/proc/self/statm", "r");
    if (NULL == fp) {
        return -1;
    }
    if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
              &m.size, &m.resident, &m.share,
              &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {
        return -1;
    }
    return m.resident * getpagesize()/1024.0/1024.0;
  }

}
namespace embedding {

class ReDistributionOpTest : public OpsTestBase {
 protected:
  void MakeFSOp() {
    TF_ASSERT_OK(NodeDefBuilder("filter_storage", "FilterStorage")
                .Input(FakeInput(DT_RESOURCE))
                .Attr("partition_id", 0)
                .Attr("new_partition_nums", 4)
                .Attr("Tkeys", DT_INT64)
                .Attr("dtype", DT_FLOAT)
                .Finalize(node_def()));
  }

  void MakeISOp() {
    TF_ASSERT_OK(NodeDefBuilder("import_storage", "ImportStorage")
                .Input(FakeInput(DT_RESOURCE))
                .Input(FakeInput(1, DT_INT64))
                .Input(FakeInput(1, DT_FLOAT))
                .Input(FakeInput(1, DT_INT64))
                .Input(FakeInput(1, DT_INT64))
                .Attr("partition_id", 0)
                .Attr("partition_nums", 1)
                .Attr("Tkeys", DT_INT64)
                .Attr("dtype", DT_FLOAT)
                .Finalize(node_def()));
  }

  void MakeReAssignOp() {
    TF_ASSERT_OK(NodeDefBuilder("re_assign", "ReAssign")
                .Input(FakeInput(MakeRefType(DT_INT64)))
                .Input(FakeInput(DT_INT64))
                .Input(FakeInput(DT_INT32))
                .Attr("partition_id", 0)
                .Attr("partition_nums", 2)
                .Attr("T", DT_INT64)
                .Finalize(node_def()));
  }
};

TEST_F(ReDistributionOpTest, TestEVFilterStorage) {
  MakeFSOp();
  TF_ASSERT_OK(InitOp());
  
  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage = embedding::StorageFactory::Create<int64, float>(
      embedding::StorageConfig(), cpu_allocator(), "EmbeddingVar");
  auto embedding_var = new EmbeddingVar<int64, float>("EmbeddingVar",
      storage, EmbeddingConfig(0, 0, 1, 1, "", 5),
      cpu_allocator());
  embedding_var->Init(value, 1);
  LOG(INFO) << "Insert EmeddingVar Keys";
  int64 ev_size = 1000000;
  for (int64 i = 0; i < ev_size; i++) {
    float *output = (float *)malloc(value_size*sizeof(float));
    embedding_var->LookupOrCreate(i, output, fill_v);
  }

  ASSERT_EQ(embedding_var->Size(), ev_size);
  AddResourceInput<EmbeddingVar<int64, float>>("", "EmbeddingVar", embedding_var);
  auto proc_mem = read_proc_memory();
  TF_ASSERT_OK(RunOpKernel());
  auto after_proc_mem = read_proc_memory();
  LOG(INFO) << "before filter mem usage: " << proc_mem << " MB"
            << " after filter mem usage: " << after_proc_mem << " MB";

  // Check the output sizes
  {  // Output 0
    std::vector<int64> unneeded_ids;
    unneeded_ids.reserve(ev_size / 4 * 3);
    for (int64 i = 0; i < ev_size; ++i) {
        if (i % 4 != 0) {
            unneeded_ids.push_back(i);
        }
    }
    Tensor expected(allocator(), DT_INT64, TensorShape({ev_size / 4 * 3}));
    test::FillValues<int64>(&expected, unneeded_ids);
    test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
  }

  // Check the output sizes
  {  // Output 0
    std::vector<float> unneeded_values;
    unneeded_values.reserve(ev_size / 4 * 3 * value_size);
    for (int64 i = 0; i < ev_size; ++i) {
        if (i % 4 != 0) {
            for (int j = 0; j < value_size; ++j) {
                unneeded_values.push_back(0.0);
            }
        }
    }
    Tensor expected(allocator(), DT_FLOAT, TensorShape({ev_size / 4 * 3, value_size}));
    test::FillValues<float>(&expected, unneeded_values);
    // test::ExpectTensorEqual<float>(expected, *GetOutput(1));

  }

  // Check the output sizes
  {  // Output 0
    std::vector<int64> unneeded_versions;
    unneeded_versions.reserve(ev_size / 4 * 3);
    for (int64 i = 0; i < ev_size; ++i) {
        if (i % 4 != 0) {
            unneeded_versions.push_back(-1);
        }
    }
    Tensor expected(allocator(), DT_INT64, TensorShape({ev_size / 4 * 3}));
    test::FillValues<int64>(&expected, unneeded_versions);
    test::ExpectTensorEqual<int64>(expected, *GetOutput(2));

  }

  {  
    std::vector<int64> unneeded_freqs;
    unneeded_freqs.reserve(ev_size / 4 * 3);
    for (int64 i = 0; i < ev_size; ++i) {
        if (i % 4 != 0) {
            unneeded_freqs.push_back(0);
        }
    }
    Tensor expected(allocator(), DT_INT64, TensorShape({ev_size / 4 * 3}));
    test::FillValues<int64>(&expected, unneeded_freqs);
    test::ExpectTensorEqual<int64>(expected, *GetOutput(3));
  }

  ASSERT_EQ(embedding_var->Size(), ev_size / 4);
  
}

TEST_F(ReDistributionOpTest, TestEVImportStorage) {
  MakeISOp();
  TF_ASSERT_OK(InitOp());

  int64 value_size = 128;
  Tensor value(DT_FLOAT, TensorShape({value_size}));
  test::FillValues<float>(&value, std::vector<float>(value_size, 9.0));
  float* fill_v = (float*)malloc(value_size * sizeof(float));
  auto storage = embedding::StorageFactory::Create<int64, float>(
      embedding::StorageConfig(), cpu_allocator(), "EmbeddingVar");
  auto embedding_var = new EmbeddingVar<int64, float>("EmbeddingVar",
      storage, EmbeddingConfig(0, 0, 1, 1, "", 5),
      cpu_allocator());
  embedding_var->Init(value, 1);
  LOG(INFO) << "Inserting EV";
  int64 ev_size = 1000000;
  for (int64 i = 1; i < ev_size; i+=2) {
    float *output = (float *)malloc(value_size*sizeof(float));
    embedding_var->LookupOrCreate(i, output, fill_v);
    delete output;
  }

  ASSERT_EQ(embedding_var->Size(), ev_size / 2);
  AddResourceInput<EmbeddingVar<int64, float>>("", "EmbeddingVar", embedding_var);
  
  {
    std::vector<int64> sp_ids_vec;
    sp_ids_vec.reserve(ev_size / 2);
    for (int64 i = 0; i < ev_size; i+=2) {
        sp_ids_vec.push_back(i);
    }
    Tensor sp_ids(DT_INT64, {ev_size / 2});
    test::FillValues<int64>(&sp_ids, sp_ids_vec);
    AddInputFromArray<int64>(sp_ids.shape(), sp_ids.flat<int64>());

    std::vector<float> values_vec;
    values_vec.reserve(ev_size / 2 * value_size);
    for (int64 i = 0; i < ev_size; i+=2) {
        for (int j = 0; j < value_size; ++j) {
            values_vec.push_back(i * 5.0);
        } 
    }
    Tensor values(DT_FLOAT, {ev_size / 2, value_size});
    test::FillValues<float>(&values, values_vec);
    AddInputFromArray<float>(values.shape(), values.flat<float>());

    std::vector<int64> versions_vec;
    versions_vec.reserve(ev_size / 2);
    for (int64 i = 0; i < ev_size; i+=2) {
        versions_vec.push_back(-1);
    }
    Tensor versions(DT_INT64, {ev_size / 2});
    test::FillValues<int64>(&versions, versions_vec);
    AddInputFromArray<int64>(versions.shape(), versions.flat<int64>());

    std::vector<int64> freqs_vec;
    freqs_vec.reserve(ev_size / 2);
    for (int64 i = 0; i < ev_size; i+=2) {
        freqs_vec.push_back(5);
    }
    Tensor freqs(DT_INT64, {ev_size / 2});
    test::FillValues<int64>(&freqs, freqs_vec);
    AddInputFromArray<int64>(freqs.shape(), freqs.flat<int64>());
  }

  auto proc_mem = read_proc_memory();
  TF_ASSERT_OK(RunOpKernel());
  auto after_proc_mem = read_proc_memory();
  LOG(INFO) << "before import mem usage: " << proc_mem << " MB"
            << " after import mem usage: " << after_proc_mem << " MB";

  ASSERT_EQ(embedding_var->Size(), ev_size);

  for (int64 i = 0; i < ev_size; i+=2) {
    float *val = (float *)malloc((value_size+1)*sizeof(float));
    float *default_value = (float *)malloc((value_size)*sizeof(float));
    for (int k = 0; k < value_size; k++) {
      default_value[k] = 10.0;
    }
    embedding_var->Lookup(i, val, default_value);
    ASSERT_EQ(val[0], i * 5.0);
    int ret_version = embedding_var->GetVersion(i);
    ASSERT_EQ(ret_version, -1);
    // int ret_freq = embedding_var->GetFreq(i);
    // ASSERT_EQ(ret_freq, 5);
    free(val);
    free(default_value);
  }
  
  // for (int64 i = 1; i < ev_size; i+=2) {
  //   float *val = (float *)malloc((value_size+1)*sizeof(float));
  //   float *default_value = (float *)malloc((value_size)*sizeof(float));
  //   for (int k = 0; k < value_size; k++) {
  //     default_value[k] = 10.0;
  //   }
  //   embedding_var->Lookup(i, val, default_value);
  //   ASSERT_EQ(val[0], 9.0);
  //   free(val);
  //   free(default_value);
  // }

}

} // namespace embedding
} // namespace tensorflow