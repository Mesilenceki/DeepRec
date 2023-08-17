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
==============================================================================*/

#include "tensorflow/core/graph/elastic_partition_pass.h"

#include "include/json/json.h"
#include "gtest/gtest.h"

#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class ElasticTrainingPassTest : public ::testing::Test {
  public:
    ElasticTrainingPassTest() {}

    static void VerifyNodes(Node* node, const std::vector<Node*>& expected_in,
                          const std::vector<Node*>& expected_out) {
    std::vector<Node*> in;
    for (const Edge* e : node->in_edges()) {
      in.push_back(e->src());
    }
    EXPECT_EQ(Stringify(expected_in), Stringify(in));

    std::vector<Node*> out;
    for (const Edge* e : node->out_edges()) {
      out.push_back(e->dst());
    }
    EXPECT_EQ(Stringify(expected_out), Stringify(out));
  }
 private:
  // Convert a list of nodes to a sorted list of strings so failure messages
  // are readable.
  static std::vector<string> Stringify(const std::vector<Node*>& nodes) {
    std::vector<string> result;
    result.reserve(nodes.size());
    for (Node* n : nodes) {
      result.push_back(n->DebugString());
    }
    std::sort(result.begin(), result.end());
    return result;
  }
 
};

void ConstructGraph(std::unique_ptr<Graph>* graph){
  // Graph:
    //       b
    //       |
    //       v
    // a -> dynamicpartition (ClusterX) -> relu0 (ClusterX) -> stage
    //
    //             b
    //             |
    //             v
    // unstage -> add1 (ClusterY) -> relu1 (ClusterY)
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* ids = ops::SourceOp("Const", builder.opts()
                                .WithName("ids")
                                .WithAttr("dtype", DT_INT64)
                                .WithAttr("value", Tensor()));
    Node* partitions = ops::SourceOp("Const", builder.opts()
                                    .WithName("partitions")
                                    .WithAttr("dtype", DT_INT32)
                                    .WithAttr("value", Tensor()));
     Node* indices = ops::SourceOp("Const", builder.opts()
                                .WithName("indices")
                                .WithAttr("dtype", DT_INT32)
                                .WithAttr("value", Tensor()));

    Node* dp = ops::BinaryOp("DynamicPartition", ids, partitions, builder.opts().WithName("DynamicPartition")
                                                                                .WithAttr("num_partitions", 2)
                                                                                .WithAttr("T", DT_INT64));
    Node* dp_1 = ops::BinaryOp("DynamicPartition", indices, partitions, builder.opts().WithName("DynamicPartition_1")
                                                                                      .WithAttr("num_partitions", 2)
                                                                                      .WithAttr("T", DT_INT32));

    EXPECT_EQ(Status::OK(), GraphDefBuilderToGraph(builder, graph->get()));
};

void InitEnv(bool enable_pass, int partition_num) {
  if (enable_pass) {
    setenv("ENABLE_ELASTIC", "1", 1);
  }
  std::string tf_config_str;
  if (partition_num == 2) {
    tf_config_str = "{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"chief\", \"index\": 0}}";
  } else {
    tf_config_str = "{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"chief\", \"index\": 0}}";
  }
   
  Json::Value tf_config;
  Json::FastWriter writer;
  std::string new_tf_config = writer.write(tf_config);
  setenv("TF_CONFIG", tf_config_str.c_str(), 1);
}

// Status ElasticTraining(std::unique_ptr<Graph>* graph) {
//   GraphOptimizationPassOptions opt_options;
//   opt_options.graph = graph;
//   FunctionDefLibrary fdef_lib;
//   FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
//   opt_options.flib_def = &flib_def;
//   SessionOptions session_options;
//   session_options.env = Env::Default();
//   opt_options.session_options = &session_options;
//   ElasticTrainingPass pass;
  
//   EXPECT_EQ(Status::OK(), pass.UpdatePartitionNums());
//   std::vector<Node*> ev_node_vec {ev_0, ev_1, ev_2};
//   EXPECT_EQ(Status::OK(), pass.RewriteElasticPartitionGraph(graph.get(), ev_node_vec));
//   // EXPECT_EQ(9 , graph->num_op_nodes());
//   return pass.Run(opt_options);
// }

TEST_F(ElasticTrainingPassTest, RewriteDynamicPartitionSubGraph) {
  InitEnv(true, 3);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  // Graph:
  //       b
  //       |
  //       v
  // a -> dynamicpartition (ClusterX) -> relu0 (ClusterX) -> stage
  //
  //             b
  //             |
  //             v
  // unstage -> add1 (ClusterY) -> relu1 (ClusterY)
  GraphDefBuilder builder;

  Node* ids = ops::SourceOp("Const", builder.opts()
                              .WithName("ids")
                              .WithAttr("dtype", DT_INT64)
                              .WithAttr("value", Tensor()));
  Node* partitions = ops::SourceOp("Const", builder.opts()
                                  .WithName("partitions")
                                  .WithAttr("dtype", DT_INT32)
                                  .WithAttr("value", Tensor()));
  Node* start = ops::SourceOp("Const", builder.opts()
                              .WithName("start")
                              .WithAttr("dtype", DT_INT32)
                              .WithAttr("value", Tensor()));

  Node* size = ops::SourceOp("Const", builder.opts()
                              .WithName("size")
                              .WithAttr("dtype", DT_INT32)
                              .WithAttr("value", Tensor()));

  Node* delta = ops::SourceOp("Const", builder.opts()
                              .WithName("delta")
                              .WithAttr("dtype", DT_INT32)
                              .WithAttr("value", Tensor()));

  NodeBuilder range_def = NodeBuilder("Range", "Range")
                                .Input(start, 0)
                                .Input(size, 0)
                                .Input(delta, 0)
                                .Attr("Tidx", DT_INT32);
  Node* range = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0").FinalizeBuilder(&range_def);

  Node* dp = ops::BinaryOp("DynamicPartition", ids, partitions, builder.opts().WithName("DynamicPartition")
                                                                              .WithAttr("num_partitions", 2)
                                                                              .WithAttr("T", DT_INT64));
  Node* dp_1 = ops::BinaryOp("DynamicPartition", range, partitions, builder.opts().WithName("DynamicPartition_1")
                                                                                    .WithAttr("num_partitions", 2)
                                                                                    .WithAttr("T", DT_INT32));

  TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(8);
  Node* ev_0 = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                            .WithName("input_layer/input_layer/C10_embedding/embedding_weights/part_0")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_0")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  Node* axis_0 = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                                .WithName("ev_0/axis")
                                .WithAttr("dtype", DT_FLOAT)
                                .WithAttr("value", Tensor()));

  NodeBuilder gather_builder_0 = NodeBuilder("ev_0/Gather", "KvResourceGather")
                                .Input(ev_0, 0)
                                .Input(dp, 0)
                                .Input(axis_0, 0)
                                .Attr("dtype", DT_FLOAT);
  Node* gather_0 = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0").FinalizeBuilder(&gather_builder_0);

  Node* identity_0 = ops::UnaryOp("Identity", gather_0, builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                                                                      .WithName("ev_0/Identity")
                                                                      .WithAttr("T", DT_FLOAT));

  Node* ev_1 = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0")
                            .WithName("input_layer/input_layer/C10_embedding/embedding_weights/part_1")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_1")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));
  
  Node* axis_1 = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0")
                                .WithName("ev_1/axis")
                                .WithAttr("dtype", DT_FLOAT)
                                .WithAttr("value", Tensor()));

  NodeBuilder gather_builder_1 = NodeBuilder("ev_1/Gather", "KvResourceGather")
                                .Input(ev_1, 0)
                                .Input(dp, 1)
                                .Input(axis_1, 0)
                                .Attr("dtype", DT_FLOAT);
  Node* gather_1 = builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0").FinalizeBuilder(&gather_builder_1);

  Node* identity_1 = ops::UnaryOp("Identity", gather_1, builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0")
                                                                      .WithName("ev_1/Identity")
                                                                      .WithAttr("T", DT_FLOAT));

  Node* ev_2 = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:2/device:CPU:0")
                            .WithName("input_layer/input_layer/C10_embedding/embedding_weights/part_2")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_2")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  Node* axis_2 = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:2/device:CPU:0")
                                .WithName("ev_2/axis")
                                .WithAttr("dtype", DT_FLOAT)
                                .WithAttr("value", Tensor()));

  NodeBuilder gather_builder_2 = NodeBuilder("ev_2/Gather", "KvResourceGather")
                                .Input(ev_2, 0)
                                .Input(ids, 0)
                                .Input(axis_2, 0)
                                .Attr("dtype", DT_FLOAT);
  Node* gather_2 = builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0").FinalizeBuilder(&gather_builder_2);

  Node* identity_2 = ops::UnaryOp("Identity", gather_2, builder.opts().WithDevice("/job:ps/replica:0/task:2/device:CPU:2")
                                                                      .WithName("ev_2/Identity")
                                                                      .WithAttr("T", DT_FLOAT));

  std::vector<NodeBuilder::NodeOut> indices_vec {NodeBuilder::NodeOut(dp_1, 0), NodeBuilder::NodeOut(dp_1, 1)};
  std::vector<NodeBuilder::NodeOut> data_vec {NodeBuilder::NodeOut(identity_0, 0), NodeBuilder::NodeOut(identity_1, 0)};

  NodeBuilder dynamic_stitch_builder = NodeBuilder("ParallelDynamicStitch", "ParallelDynamicStitch")
                                .Input(indices_vec)
                                .Input(data_vec)
                                .Attr("N", 2)
                                .Attr("T", DT_FLOAT);
  Node* dynamic_stitch = builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0").FinalizeBuilder(&dynamic_stitch_builder);

  EXPECT_EQ(Status::OK(), GraphDefBuilderToGraph(builder, graph.get()));
  ElasticTrainingPass pass;
  
  EXPECT_EQ(Status::OK(), pass.UpdatePartitionNums());
  std::vector<Node*> ev_node_vec {graph->FindNodeId(ev_0->id()), graph->FindNodeId(ev_1->id()), graph->FindNodeId(ev_2->id())};
  EXPECT_EQ(Status::OK(), pass.RewriteElasticPartitionGraph(graph.get(), ev_node_vec));

  // EXPECT_EQ(3, graph->num_op_nodes());
  // LOG(INFO) << graph->ToGraphDefDebug().DebugString();
}


TEST_F(ElasticTrainingPassTest, AddEmbeddingSubGraph) {
  InitEnv(true, 2);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(8);
  Node* primary_ev = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                            .WithName("input_layer/input_layer/C12_embedding/embedding_weights/part_0")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_0")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  Node* ev = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                            .WithName("input_layer/input_layer/C12_embedding/embedding_weights/part_0/AdamAsync_2_sparse_incr")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_0/AdamAsync_2_sparse_incr")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  Node* value = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                .WithName("value")
                                .WithAttr("dtype", DT_FLOAT)
                                .WithAttr("value", Tensor()));
  
  Node* empty_key = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                .WithName("empty_key")
                                .WithAttr("dtype", DT_INT64)
                                .WithAttr("value", Tensor()));

  NodeBuilder init_ev_builder = NodeBuilder("ev_init_op", "InitializeKvVariableOp")
                        .Input(ev, 0)
                        .Input(primary_ev, 0)
                        .Input(value, 0)
                        .Input(empty_key, 0)
                        .Attr("Tkeys", DT_INT64)
                        .Attr("dtype", DT_FLOAT)
                        .Attr("counter_type", DT_FLOAT)
                        .Attr("shape", tshape_proto);
  Node* init_ev = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").FinalizeBuilder(&init_ev_builder);

  Node* indices = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                .WithName("ids")
                                .WithAttr("dtype", DT_INT64)
                                .WithAttr("value", Tensor()));

  Node* axis = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                .WithName("axis")
                                .WithAttr("dtype", DT_INT32)
                                .WithAttr("value", Tensor()));
  NodeBuilder gather_builder = NodeBuilder("Gather", "GatherV2")
                                .Input(primary_ev, 0)
                                .Input(indices, 0)
                                .Input(axis, 0);
  Node* gather = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").FinalizeBuilder(&gather_builder);
  Node* identity = ops::UnaryOp("Identity", gather, builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").WithName("Identity"));

  EXPECT_EQ(Status::OK(), GraphDefBuilderToGraph(builder, graph.get()));
  graph->set_assigned_device_name(ev, "/job:ps/replica:0/task:0/device:CPU:2");
  graph->set_assigned_device_name(primary_ev, "/job:ps/replica:0/task:0/device:CPU:2");
  int num_origin_nodes = graph->num_op_nodes();
  ElasticTrainingPass pass;
  EXPECT_EQ(Status::OK(), pass.UpdatePartitionNums());
  // EXPECT_EQ(Status::OK(), pass.RewriteTrainingGraph(graph.get(), true));
  // EXPECT_EQ(2*num_origin_nodes , graph->num_op_nodes());
  // LOG(INFO) << graph->ToGraphDefDebug().DebugString();
}

TEST_F(ElasticTrainingPassTest, RemoveEmbeddingSubGraph) {
  InitEnv(true, 2);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(8);
  for (int i = 0; i < 3; ++i) {
    std::string primary_ev_name = "input_layer/input_layer/C12_embedding/embedding_weights/part_" + std::to_string(i);
    Node* primary_ev = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                            .WithName(primary_ev_name)
                            .WithAttr("container", "")
                            .WithAttr("shared_name", primary_ev_name)
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

    std::string ev_name = "input_layer/input_layer/C12_embedding/embedding_weights/part_" + std::to_string(i) +"/AdamAsync_2_sparse_incr";
    Node* ev = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                              .WithName(ev_name)
                              .WithAttr("container", "")
                              .WithAttr("shared_name", ev_name)
                              .WithAttr("dtype", DT_FLOAT)
                              .WithAttr("shape", tshape_proto)
                              .WithAttr("Tkeys", DT_INT64));

    Node* value = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("ev_" + std::to_string(i) + "/value")
                                  .WithAttr("dtype", DT_FLOAT)
                                  .WithAttr("value", Tensor()));
    
    Node* empty_key = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("ev_" + std::to_string(i) + "/empty_key")
                                  .WithAttr("dtype", DT_INT64)
                                  .WithAttr("value", Tensor()));

    NodeBuilder init_ev_builder = NodeBuilder("ev_" + std::to_string(i) +"/ev_init_op", "InitializeKvVariableOp")
                          .Input(ev, 0)
                          .Input(primary_ev, 0)
                          .Input(value, 0)
                          .Input(empty_key, 0)
                          .Attr("Tkeys", DT_INT64)
                          .Attr("dtype", DT_FLOAT)
                          .Attr("counter_type", DT_FLOAT)
                          .Attr("shape", tshape_proto);
    Node* init_ev = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").FinalizeBuilder(&init_ev_builder);

    Node* indices = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("ev_" + std::to_string(i) + "/ids")
                                  .WithAttr("dtype", DT_INT64)
                                  .WithAttr("value", Tensor()));

    Node* axis = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("ev_" + std::to_string(i) + "/axis")
                                  .WithAttr("dtype", DT_INT32)
                                  .WithAttr("value", Tensor()));
    NodeBuilder gather_builder = NodeBuilder("ev_" + std::to_string(i) + "/Gather", "GatherV2")
                                  .Input(primary_ev, 0)
                                  .Input(indices, 0)
                                  .Input(axis, 0);
    Node* gather = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").FinalizeBuilder(&gather_builder);
    Node* identity = ops::UnaryOp("Identity", gather, builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").WithName("ev_" + std::to_string(i) + "/Identity"));
  }

  EXPECT_EQ(Status::OK(), GraphDefBuilderToGraph(builder, graph.get()));
  // LOG(INFO) << graph->ToGraphDefDebug().DebugString();
  int num_origin_nodes = graph->num_op_nodes();
  ElasticTrainingPass pass;
  EXPECT_EQ(Status::OK(), pass.UpdatePartitionNums());
  // EXPECT_EQ(Status::OK(), pass.RewriteTrainingGraph(graph.get(), true));
  // LOG(INFO) << graph->ToGraphDefDebug().DebugString();
  EXPECT_EQ(num_origin_nodes / 3 * 2 , graph->num_op_nodes() - 1/* redundant 1 is the gather ids*/);
}

// ScalingUpRewriteImportStorage
TEST_F(ElasticTrainingPassTest, ScalingUpRedistributionGraph) {
  InitEnv(true, 3);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(8);
  Node* ev_0 = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                            .WithName("input_layer/input_layer/C10_embedding/embedding_weights/part_0")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_0")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  Node* filter_storage_op_0 = ops::UnaryOp("FilterStorage", ev_0, builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                                                                                  .WithName("ev_0/FilterStorage")
                                                                                  .WithAttr("Tkeys", DT_INT64)
                                                                                  .WithAttr("dtype", DT_FLOAT));
  
  Node* ev_1 = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0")
                            .WithName("input_layer/input_layer/C10_embedding/embedding_weights/part_1")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_1")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  Node* filter_storage_op_1 = ops::UnaryOp("FilterStorage", ev_1, builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0")
                                                                                  .WithName("ev_1/FilterStorage")
                                                                                  .WithAttr("Tkeys", DT_INT64)
                                                                                  .WithAttr("dtype", DT_FLOAT));

  NodeBuilder ev_0_import_storage_builder = NodeBuilder("ev_0/ImportStorage", "ImportStorage")
                                                      .Input(ev_0, 0)
                                                      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(filter_storage_op_1, 0)})
                                                      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(filter_storage_op_1, 1)})
                                                      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(filter_storage_op_1, 2)})
                                                      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(filter_storage_op_1, 3)})
                                                      .Attr("Tkeys", DT_INT64)
                                                      .Attr("dtype", DT_FLOAT)
                                                      .Attr("partition_id", 0)
                                                      .Attr("partition_nums", 1);
  Node* ev_0_import_storage = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0").FinalizeBuilder(&ev_0_import_storage_builder);

  NodeBuilder ev_1_import_storage_builder = NodeBuilder("ev_1/ImportStorage", "ImportStorage")
                                                      .Input(ev_1, 0)
                                                      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(filter_storage_op_0, 0)})
                                                      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(filter_storage_op_0, 1)})
                                                      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(filter_storage_op_0, 2)})
                                                      .Input(std::vector<NodeBuilder::NodeOut>{NodeBuilder::NodeOut(filter_storage_op_0, 3)})
                                                      .Attr("Tkeys", DT_INT64)
                                                      .Attr("dtype", DT_FLOAT)
                                                      .Attr("partition_id", 1)
                                                      .Attr("partition_nums", 1);
  Node* ev_1_import_storage = builder.opts().WithDevice("/job:ps/replica:0/task:1/device:CPU:0").FinalizeBuilder(&ev_1_import_storage_builder);

  Node* ev_2 = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:2/device:CPU:0")
                            .WithName("input_layer/input_layer/C10_embedding/embedding_weights/part_2")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_2")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  EXPECT_EQ(Status::OK(), GraphDefBuilderToGraph(builder, graph.get()));
  EXPECT_EQ(7 , graph->num_op_nodes());
  ElasticTrainingPass pass;
  
  EXPECT_EQ(Status::OK(), pass.UpdatePartitionNums());
  std::vector<Node*> ev_node_vec {ev_0, ev_1, ev_2};
  EXPECT_EQ(Status::OK(), pass.ScalingUpRedistributionGraph(graph.get(), ev_node_vec, 2));
  EXPECT_EQ(9 , graph->num_op_nodes());
  // LOG(INFO) << graph->ToGraphDefDebug().DebugString();
}


// ScalingDownRewriteFilterStorage
TEST_F(ElasticTrainingPassTest, ScalingUpSaveSubGraph) {
  InitEnv(true, 2);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(8);
  Node* ev_0 = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                            .WithName("input_layer/input_layer/C10_embedding/embedding_weights/part_0")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_0")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  Node* kv_lookup_resource_0 = ops::UnaryOp("KvResourceLookupResource", ev_0, builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                                                                                  .WithName("ev_0/KvResourceLookupResource")
                                                                                  .WithAttr("Tkeys", DT_INT64)
                                                                                  .WithAttr("dtype", DT_FLOAT));
  
  Node* ev_1 = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                            .WithName("input_layer/input_layer/C12_embedding/embedding_weights/part_0")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_1")
                            .WithAttr("dtype", DT_FLOAT)
                            .WithAttr("shape", tshape_proto)
                            .WithAttr("Tkeys", DT_INT64));

  Node* kv_lookup_resource_1 = ops::UnaryOp("KvResourceLookupResource", ev_1, builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                                                                                  .WithName("ev_1/KvResourceLookupResource")
                                                                                  .WithAttr("Tkeys", DT_INT64)
                                                                                  .WithAttr("dtype", DT_FLOAT));
  Tensor t_filename(DT_STRING, TensorShape({}));
  t_filename.scalar<tstring>()() = "fsadfasdf";
  Node* filename = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("filename")
                                  .WithAttr("dtype", DT_STRING)
                                  .WithAttr("value", t_filename));

  Node* tmpname = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("tmpname")
                                  .WithAttr("dtype", DT_STRING)
                                  .WithAttr("value", t_filename));
  
  
  std::vector<NodeBuilder::NodeOut> string_input{NodeBuilder::NodeOut(filename, 0), NodeBuilder::NodeOut(tmpname, 0)};
  NodeBuilder string_join_builder = NodeBuilder("StringJoin", "StringJoin")
                                              .Input(string_input)
                                              .Attr("N", 2);
  Node* string_join = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0").FinalizeBuilder(&string_join_builder);

  Tensor t_shard_id(DT_INT32, TensorShape({}));
  t_shard_id.scalar<int32>()() = 0;
  Node* shard_id = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("shard")
                                  .WithAttr("dtype", DT_INT32)
                                  .WithAttr("value", t_shard_id));
  Tensor t_shard_num(DT_INT32, TensorShape({}));
  t_shard_num.scalar<int32>()() = 1;
  Node* shard_num = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("num_shards")
                                  .WithAttr("dtype", DT_INT32)
                                  .WithAttr("value", t_shard_num));

  NodeBuilder sharded_filename_builder = NodeBuilder("ShardedFilename", "ShardedFilename")
                                              .Input(string_join, 0)
                                              .Input(shard_id, 0)
                                              .Input(shard_num, 0);
  Node* sharded_filename = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0").FinalizeBuilder(&sharded_filename_builder);
  TensorShapeProto t_proto;
  t_proto.add_dim()->set_size(0);
  Node* variable = ops::SourceOp("VariableV2", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0")
                            .WithName("global_step")
                            .WithAttr("container", "")
                            .WithAttr("shared_name", "global_step")
                            .WithAttr("shape", t_proto)
                            .WithAttr("dtype", DT_INT64));

  Node* global_step = ops::UnaryOp("Identity", variable, builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0").WithName("global_step/read"));
  
  Tensor t_global_step(DT_STRING, TensorShape({}));
  t_global_step.scalar<tstring>()() = "global_step";
  Node* tensor_name = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("tensor_name")
                                  .WithAttr("dtype", DT_STRING)
                                  .WithAttr("value", t_global_step));

  Tensor t_ev_name(DT_STRING, TensorShape({2}));
  t_ev_name.flat<tstring>()(0) = "input_layer/input_layer/C10_embedding/embedding_weights/part_0";
  t_ev_name.flat<tstring>()(1) = "input_layer/input_layer/C12_embedding/embedding_weights/part_0";
  Node* ev_name = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("ev_name")
                                  .WithAttr("dtype", DT_STRING)
                                  .WithAttr("value", t_ev_name));

  Node* shape_and_slice = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
                                  .WithName("shape_and_slice")
                                  .WithAttr("dtype", DT_STRING)
                                  .WithAttr("value", Tensor(DT_STRING, TensorShape({}))));

  std::vector<NodeBuilder::NodeOut> ev_resource_input{NodeBuilder::NodeOut(kv_lookup_resource_0, 0), NodeBuilder::NodeOut(kv_lookup_resource_1, 0)};
  int n = ev_resource_input.size();
  NodeBuilder ev_resource_builder = NodeBuilder("ev_resource", "Pack")
                                              .Input(ev_resource_input)
                                              .Attr("N", n)
                                              .Attr("T", DT_INT64)
                                              .Attr("axis", 0);
  Node* ev_resource = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0").FinalizeBuilder(&ev_resource_builder);

  std::vector<NodeBuilder::NodeOut> tensor_input{NodeBuilder::NodeOut(global_step, 0)};
  NodeBuilder save_builder = NodeBuilder("SaveV3", "SaveV3")
                                        .Input(sharded_filename, 0)
                                        .Input(tensor_name, 0)
                                        .Input(shape_and_slice, 0)
                                        .Input(ev_name, 0)
                                        .Input(ev_resource, 0)
                                        .Input(tensor_input)
                                        .Attr("dtypes", {DT_INT64})
                                        .Attr("ev_key_types", {DT_INT64, DT_INT64})
                                        .Attr("has_ev", true);
  Node* save_op = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:0").FinalizeBuilder(&save_builder);

  EXPECT_EQ(Status::OK(), GraphDefBuilderToGraph(builder, graph.get()));
  LOG(INFO) << graph->ToGraphDefDebug().DebugString();
  // EXPECT_EQ(9 , graph->num_op_nodes());
  ElasticTrainingPass pass;
  
  EXPECT_EQ(Status::OK(), pass.UpdatePartitionNums());
  EXPECT_EQ(Status::OK(), pass.RewriteTrainingGraph(graph.get(), true));
  LOG(INFO) << graph->ToGraphDefDebug().DebugString();
  // EXPECT_EQ(7 , graph->num_op_nodes());
}

// TEST_F(ElasticTrainingPassTest, AddEmbeddingSubGraph) {
//   InitEnv(true, 2);
//   std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
//   GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
//   TensorShapeProto tshape_proto;
//   tshape_proto.add_dim()->set_size(8);
//   Node* primary_ev = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
//                             .WithName("input_layer/input_layer/C12_embedding/embedding_weights/part_0")
//                             .WithAttr("container", "")
//                             .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_0")
//                             .WithAttr("dtype", DT_FLOAT)
//                             .WithAttr("shape", tshape_proto)
//                             .WithAttr("Tkeys", DT_INT64));

//   Node* ev = ops::SourceOp("KvVarHandleOp", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
//                             .WithName("input_layer/input_layer/C12_embedding/embedding_weights/part_0/AdamAsync_2_sparse_incr")
//                             .WithAttr("container", "")
//                             .WithAttr("shared_name", "input_layer/input_layer/C12_embedding/embedding_weights/part_0/AdamAsync_2_sparse_incr")
//                             .WithAttr("dtype", DT_FLOAT)
//                             .WithAttr("shape", tshape_proto)
//                             .WithAttr("Tkeys", DT_INT64));

//   Node* value = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
//                                 .WithName("value")
//                                 .WithAttr("dtype", DT_FLOAT)
//                                 .WithAttr("value", Tensor()));
  
//   Node* empty_key = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
//                                 .WithName("empty_key")
//                                 .WithAttr("dtype", DT_INT64)
//                                 .WithAttr("value", Tensor()));

//   NodeBuilder init_ev_builder = NodeBuilder("ev_init_op", "InitializeKvVariableOp")
//                         .Input(ev, 0)
//                         .Input(primary_ev, 0)
//                         .Input(value, 0)
//                         .Input(empty_key, 0)
//                         .Attr("Tkeys", DT_INT64)
//                         .Attr("dtype", DT_FLOAT)
//                         .Attr("counter_type", DT_FLOAT)
//                         .Attr("shape", tshape_proto);
//   Node* init_ev = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").FinalizeBuilder(&init_ev_builder);

//   Node* indices = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
//                                 .WithName("ids")
//                                 .WithAttr("dtype", DT_INT64)
//                                 .WithAttr("value", Tensor()));

//   Node* axis = ops::SourceOp("Const", builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2")
//                                 .WithName("axis")
//                                 .WithAttr("dtype", DT_INT32)
//                                 .WithAttr("value", Tensor()));
//   NodeBuilder gather_builder = NodeBuilder("Gather", "GatherV2")
//                                 .Input(primary_ev, 0)
//                                 .Input(indices, 0)
//                                 .Input(axis, 0);
//   Node* gather = builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").FinalizeBuilder(&gather_builder);
//   Node* identity = ops::UnaryOp("Identity", gather, builder.opts().WithDevice("/job:ps/replica:0/task:0/device:CPU:2").WithName("Identity"));

//   EXPECT_EQ(Status::OK(), GraphDefBuilderToGraph(builder, graph.get()));
//   graph->set_assigned_device_name(ev, "/job:ps/replica:0/task:0/device:CPU:2");
//   graph->set_assigned_device_name(primary_ev, "/job:ps/replica:0/task:0/device:CPU:2");
//   int num_origin_nodes = graph->num_op_nodes();
//   ElasticTrainingPass pass;
//   EXPECT_EQ(Status::OK(), pass.UpdatePartitionNums());
//   EXPECT_EQ(Status::OK(), pass.RewriteTrainingGraph(graph.get(), true));
//   // EXPECT_EQ(2*num_origin_nodes , graph->num_op_nodes());
//   // LOG(INFO) << graph->ToGraphDefDebug().DebugString();
// }


} // namespace tensorflow