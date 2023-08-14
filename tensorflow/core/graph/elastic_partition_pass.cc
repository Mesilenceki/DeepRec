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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"

constexpr char kDynamicPartition[] = "DynamicPartition";
constexpr char kPart[] = "part_";
namespace tensorflow {

Status ElasticTrainingPass::Run(const GraphOptimizationPassOptions& options) {
  bool enable_elastic_training = false;
  TF_RETURN_IF_ERROR(ReadBoolFromEnvVar("ENABLE_ELASTIC", 
                                        false, &enable_elastic_training));
  
  TF_RETURN_IF_ERROR(UpdatePartitionNums());
  Graph *graph = options.graph->get();
  if (graph == nullptr)
    return errors::Internal("a graph should be available.");

  std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
  CopyGraph(*graph, new_graph.get());

  TF_RETURN_IF_ERROR(ElasticTrainingGraph(new_graph.get()));

  options.graph->swap(new_graph);
  return Status::OK();
}

Status ElasticTrainingPass::UpdatePartitionNums() {
  std::string tf_config;
  ReadStringFromEnvVar("TF_CONFIG", "", &tf_config);

  Json::Reader reader;
  Json::Value json_tf_config;
  if(!reader.parse(tf_config, json_tf_config)) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "PARSE TF_CONFIG ERROR");
  }

  if ((json_tf_config["cluster"].isNull()) ||
      (json_tf_config["cluster"]["ps"].isNull())) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "PARSE ps ERROR");
  }
  Json::Value ps_array = json_tf_config["cluster"]["ps"];
  partition_nums_ = ps_array.size();
  VLOG(1) << " partition_nums_ is " << partition_nums_;
  return Status::OK();
}

Status ElasticTrainingPass::ElasticTrainingGraph(Graph* g) {
  TF_RETURN_IF_ERROR(AddEmbeddingSubGraph(g));
  return Status::OK();
}

Status ElasticTrainingPass::AddEmbeddingSubGraph(Graph* g, bool is_test) {
  std::unordered_map<std::string, Node*> ev_nodes_map;
  std::unordered_map<std::string, std::pair<bool, int>> ev_metas_map;
  std::unordered_map<std::string, std::vector<Node*>> ev_to_origin_map;
  TF_RETURN_IF_ERROR(InitEVMeta(g, ev_nodes_map, ev_metas_map, ev_to_origin_map));
  TF_RETURN_IF_ERROR(InitNewPartitionSubGraph(g, ev_metas_map, ev_nodes_map, ev_to_origin_map, is_test));
  return Status::OK();
}

Status ElasticTrainingPass::InitEVMeta(Graph* g, std::unordered_map<std::string, Node* >& ev_nodes_map,
                                        std::unordered_map<std::string, std::pair<bool, int>>& ev_metas_map,
                                        std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map) {
  for (auto* node: g->op_nodes()) {
    if (node->IsKvVarHandle()) {
      ev_nodes_map.emplace(node->name(), node);
    }
  }

  for (auto& it: ev_nodes_map) {
    auto part_idx = it.first.find(kPart);
    std::string pre_str = it.first.substr(0, part_idx-1);
    std::string post_str = it.first.substr(part_idx+strlen(kPart));
    auto post_idx = post_str.find("/");
    if (post_idx == string::npos) {
      int ev_idx = std::stoi(post_str);
      if (ev_metas_map.find(pre_str) == ev_metas_map.end()) {
        ev_metas_map.emplace(pre_str, std::pair<bool, int>(true/*primary*/, 1));
        std::vector<Node*> ev_vec(partition_nums_, nullptr);
        ev_vec[ev_idx] = it.second;
        ev_to_origin_map.emplace(pre_str, std::move(ev_vec));
      } else {
        ev_metas_map[pre_str].second++;
        ev_to_origin_map[pre_str][ev_idx] = it.second;
      }
    } else {
      int ev_idx = std::stoi(post_str.substr(0, post_idx));
      string opt_name = pre_str + post_str.substr(post_idx);
      if (ev_metas_map.find(opt_name) == ev_metas_map.end()) {
        ev_metas_map.emplace(opt_name, std::pair<bool, int>(false/*optimizer*/, 1));
        std::vector<Node*> ev_vec(partition_nums_, nullptr);
        ev_vec[ev_idx] = it.second;
        ev_to_origin_map.emplace(opt_name, std::move(ev_vec));
      } else {
        ev_metas_map[opt_name].second++;
        ev_to_origin_map[opt_name][ev_idx] = it.second;
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::InitNewPartitionSubGraph(Graph* g,
                                std::unordered_map<std::string, std::pair<bool, int>>& ev_metas_map,
                                std::unordered_map<std::string, Node*>& ev_nodes_map,
                                std::unordered_map<std::string, std::vector<Node*>> ev_to_origin_map,
                                bool is_test) {
  std::unordered_map<std::string, Node*> ev_to_primary_map;
  for (auto it : ev_metas_map) {
    auto ori_ev_name = it.first;
    int ev_partition_num = it.second.second;
    if (ev_partition_num < partition_nums_) {
      for (int i = ev_partition_num; i < partition_nums_; ++i) {
        auto ev_node = ev_to_origin_map[ori_ev_name][0];
        std::string op_name;
        bool is_primary = false;
        if (std::get<0>(it.second)) {
          op_name = ori_ev_name + "/" + kPart + std::to_string(i);
          is_primary = true;
        } else {
          auto sep_idx = ori_ev_name.rfind("/");
          op_name = ori_ev_name.substr(0, sep_idx) + "/" + kPart + std::to_string(i) + ori_ev_name.substr(sep_idx);
        }
        
        auto device_name = /*ev_node->assigned_device_name()*/ std::string("/job:ps/replica:0/task:0/device:CPU:2");
        LOG(INFO) << " =============== " << op_name << " ----- " << ori_ev_name;
        std::string task_str = "task:";
        auto idx_begin = device_name.rfind(task_str);
        auto idx_end = device_name.find("device:", idx_begin);
        std::string new_device_name = 
            device_name.substr(0, idx_begin+task_str.size()) + std::to_string(i) + device_name.substr(idx_end-1);
        
        // EVHandler
        Node* new_ev_node = g->CopyNode(ev_node);
        new_ev_node->set_name(op_name);
        // shared_name
        new_ev_node->set_assigned_device_name(new_device_name);

        string primary_ev_name;
        if (is_primary) {
          ev_to_primary_map.emplace(ori_ev_name, new_ev_node);
          primary_ev_name = ori_ev_name;
        } else {
          auto sep_idx = ori_ev_name.rfind("/");
          primary_ev_name = ori_ev_name.substr(0, sep_idx);
        }
        LOG(INFO) << primary_ev_name << " --------- ";
        ev_to_origin_map[ori_ev_name][i] = new_ev_node;
        LOG(INFO) << "Rewriting InitializeEVResource --------- ";

        // InitializeEVResource
        for (auto* o_node:  ev_node->out_nodes()) {
          if (o_node->type_string() == "InitializeKvVariableOp") {
            auto* init_node = g->CopyNode(o_node);
            init_node->set_name(o_node->name() + "/Copy");
            init_node->set_assigned_device_name(new_device_name);
            g->AddEdge(new_ev_node, 0, init_node, 0);

            //primary_ev
            if (ev_to_primary_map.count(ori_ev_name) == 0){
              LOG(ERROR) << "BUG, ev should have primary ev node";
            }
            LOG(INFO) << "Step 1 --------- ";
            auto* primary_node = ev_to_primary_map[primary_ev_name];
            g->AddEdge(primary_node, 0, init_node, 1);
            //init_value
            LOG(INFO) << "Step 2 --------- ";
            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
            auto* init_value_node = g->CopyNode(init_value_edge->src());
            init_value_node->set_name(init_value_node->name() + "/Copy");
            init_value_node->set_assigned_device_name(new_device_name);
            g->AddEdge(init_value_node, init_value_edge->src_output(), init_node, 2);
            LOG(INFO) << "Step 3 --------- ";
            //empty_key
            const Edge* empty_key_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
            auto* empty_key_node = g->CopyNode(empty_key_edge->src());
            empty_key_node->set_name(empty_key_node->name() + "/Copy");
            empty_key_node->set_assigned_device_name(new_device_name);
            g->AddEdge(empty_key_node, empty_key_edge->src_output(), init_node, 3);
          }
        }
        LOG(INFO) << "Rewriting Gather --------- ";

        // Gather
        for (auto* o_node: ev_node->out_nodes()) {
          if (o_node->type_string() == "GatherV2") {
            Node* gather_op = g->CopyNode(o_node);
            gather_op->set_name(o_node->name() + "/Copy");
            gather_op->set_assigned_device_name(new_device_name);
            LOG(INFO) << "Step 4 --------- ";
            g->AddEdge(new_ev_node, 0, gather_op, 0);
            const Edge* gather_id_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(1, &gather_id_edge));
            Node* gather_id = g->CopyNode(gather_id_edge->src());
            gather_id->set_name(gather_id_edge->src()->name() + "/Copy");
            gather_id->set_assigned_device_name(new_device_name);
            LOG(INFO) << "Step 5 --------- ";
            const Edge* axis_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &axis_edge));
            Node* axis = g->CopyNode(axis_edge->src());
            axis->set_name(axis_edge->src()->name() + "/Copy");
            axis->set_assigned_device_name(new_device_name);
            for (auto* o_edge: gather_op->out_edges()) {
              if (o_edge->dst()->type_string() == "Identity") {
                Node* identity_op = g->CopyNode(o_edge->dst());
                identity_op->set_name(o_edge->dst()->name() + "/Copy");
                identity_op->set_assigned_device_name(new_device_name);
              }
            }
            LOG(INFO) << "Step 6 --------- ";
          }
        }
      }
    } else if (ev_partition_num > partition_nums_) {

    }
    auto ev_vec = ev_to_origin_map[ori_ev_name];
    if (!is_test) {
      TF_RETURN_IF_ERROR(RewriteRedistributionGraph(g, ev_vec, ev_partition_num));
      TF_RETURN_IF_ERROR(RewriteElasticPartitionGraph(g, ev_vec));
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::RewriteElasticPartitionGraph(Graph* g,
                                                         std::vector<Node*>& ev_node_vec) {
  Status s;
  Node* ev_node = ev_node_vec[0];
  Node* dynamic_partition_node = nullptr;
  for (auto* o_node : ev_node->out_nodes()) {
    if (o_node->type_string() == "GatherV2") {
      const Edge* input_edge = nullptr;
      TF_RETURN_IF_ERROR(o_node->input_edge(1, &input_edge));
      dynamic_partition_node = input_edge->src();
    }
  }

  if (dynamic_partition_node->type_string() == kDynamicPartition) {
    return Status::OK();
  }

  std::string node_name = dynamic_partition_node->name();
  int num_partitions;
  TF_RETURN_IF_ERROR(GetNodeAttr(dynamic_partition_node->attrs(), "num_partitions", &num_partitions));
  DataType key_type;
  TF_RETURN_IF_ERROR(GetNodeAttr(dynamic_partition_node->attrs(), "T", &key_type));
  const Node* a_copy;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_node(0, &a_copy));
  const Node* b_copy;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_node(1, &b_copy));
  auto idx = node_name.find(kDynamicPartition);
  std::string pre_node_name = node_name.substr(0, idx);
  
  NodeDef elastic_node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder(node_name + "/ElasticPartition", "ElasticPartition")
                                    .Input(a_copy->name(), 0, a_copy->output_type(0))
                                    .Input(b_copy->name(), 0, b_copy->output_type(0))
                                    .Attr("num_partitions", partition_nums_)
                                    .Attr("TKey", key_type)
                                    .Finalize(&elastic_node_def)); 
  
  Node* elastic_node = g->AddNode(elastic_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  std::vector<Node*> delete_nodes;

  const Edge* input_edge = nullptr;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_edge(1, &input_edge));
  for (auto* o_node: input_edge->src()->out_nodes()) {
    if (o_node->type_string() == kDynamicPartition) {
      const Edge* data_input_edge = nullptr;
      TF_RETURN_IF_ERROR(dynamic_partition_node->input_edge(0, &data_input_edge));
      if (data_input_edge->src()->type_string() != "Range") { // ID
        //Input
        elastic_node->set_name(pre_node_name + "/ElasticPartition");

        g->UpdateEdge(data_input_edge->src(), input_edge->src_output(), elastic_node, 0);
        //TODO(JUNQI): Output !!!!!!!!
        // for (auto* o_edge: o_node->out_edges()) {
        //   g->AddEdge(elastic_node, o_edge->src_output(), o_edge->dst(), o_edge->dst_input());
        // }
        delete_nodes.emplace_back(o_node);

        for (int i = 0 ;i < ev_node_vec.size(); ++i) {
          Node* ev_node = ev_node_vec[i];
          //Add Input ElasticPartition
          for (auto* o_node : ev_node->out_nodes()) {
            if (o_node->type_string() == "GatherV2") {
              g->AddEdge(elastic_node, i, o_node, 1);
            }
          }
        }
      } else { // Indices
        //Input
        elastic_node->set_name(pre_node_name + "/ElasticPartition_1");
        g->UpdateEdge(data_input_edge->src(), input_edge->src_output(), elastic_node, 1);
        //Output
        // for (auto* o_edge: o_node->out_edges()) {
        //   g->AddEdge(elastic_node, partition_nums_ + o_edge->src_output(), o_edge->dst(), o_edge->dst_input());
        // }
        delete_nodes.emplace_back(o_node);
        
        for (auto* oo_node: dynamic_partition_node->out_nodes()) {
          if (oo_node->type_string() == "DynamicStitch") {
            for (int i = 0 ;i < ev_node_vec.size(); ++i) {
              g->AddEdge(elastic_node, partition_nums_+i, oo_node, i);
            }
          }
        }
      }
    }
  }
  delete_nodes.emplace_back(input_edge->src());

  for (auto* n: delete_nodes) { g->RemoveNode(n); }
  return s;
}

Status ElasticTrainingPass::RewriteRedistributionGraph(Graph* g,
                                                       std::vector<Node*>& ev_node_vec,
                                                       int ev_partition_num) {
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  filtered_node_vec.reserve(ev_node_vec.size());
  for (int i = 0 ; i < ev_node_vec.size(); ++i) {
    auto* ev_node = ev_node_vec[i];
    
    if (i < ev_partition_num) {
      for (auto* o_node: ev_node->out_nodes()) {
        if (o_node->type_string() == "FilterStorageOp") {
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "dtype", &value_type));
          o_node->ClearAttr("new_partition_nums");
          o_node->AddAttr("new_partition_nums", partition_nums_);
          filtered_node_vec.push_back(o_node);
        }
      }
    } else {
      NodeDef filter_storage_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(ev_node->name() + "/FilterStorageOp", "FilterStorageOp")
                                        .Input(ev_node->name(), 0, ev_node->output_type(0))
                                        .Attr("partition_id", i)
                                        .Attr("new_partition_nums", partition_nums_)
                                        .Attr("Tkeys", key_type)
                                        .Attr("dtype", value_type)
                                        .Finalize(&filter_storage_node_def)); 
      Node* filter_node = g->AddNode(filter_storage_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      filtered_node_vec.push_back(filter_node);
    }
  }

  std::vector<Node*> delete_node_vec;
  for (int i = 0; i < ev_node_vec.size(); ++i) {
    auto* ev_node = ev_node_vec[i];
    
    if (i < ev_partition_num) {
      for (auto* o_node: ev_node->out_nodes()) {
        if (o_node->type_string() == "ImportStorageOp") {
          delete_node_vec.emplace_back(o_node);
        }
      }
    }
    std::vector<NodeDefBuilder::NodeOut> import_keys;
    std::vector<NodeDefBuilder::NodeOut> import_values;
    std::vector<NodeDefBuilder::NodeOut> import_versions;
    std::vector<NodeDefBuilder::NodeOut> import_freqs;
    for (int j = 0; j < filtered_node_vec.size(); ++j) {
      if (j != i) {
        import_keys.emplace_back(filtered_node_vec[j]->name(), 0, filtered_node_vec[j]->output_type(0));
        import_values.emplace_back(filtered_node_vec[j]->name(), 1, filtered_node_vec[j]->output_type(1));
        import_versions.emplace_back(filtered_node_vec[j]->name(), 2, filtered_node_vec[j]->output_type(2));
        import_freqs.emplace_back(filtered_node_vec[j]->name(), 3, filtered_node_vec[j]->output_type(3));
      }
    }

    NodeDef import_storage_node_def;
    TF_RETURN_IF_ERROR(NodeDefBuilder(ev_node->name() + "/ImportStorageOp", "ImportStorageOp")
                                      .Input(ev_node->name(), 0, ev_node->output_type(0))
                                      .Input(import_keys)
                                      .Input(import_values)
                                      .Input(import_versions)
                                      .Input(import_freqs)
                                      .Attr("partition_id", i)
                                      .Attr("partition_nums", partition_nums_-1)
                                      .Attr("Tkeys", key_type)
                                      .Attr("dtype", value_type)
                                      .Finalize(&import_storage_node_def)); 
    Node* import_node = g->AddNode(import_storage_node_def, &s);
    TF_RETURN_IF_ERROR(s);
  }

  for (auto* node: delete_node_vec) { g->RemoveNode(node);}
  return Status::OK();
}


REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 24, ElasticTrainingPass);

} // namespace tensorflow