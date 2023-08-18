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
#include "tensorflow/core/graph/algorithm.h"

constexpr char kDynamicPartition[] = "DynamicPartition";
constexpr char kPart[] = "part_";
namespace tensorflow {

Status ElasticTrainingPass::Run(const GraphOptimizationPassOptions& options) {
  bool enable_elastic_training = false;
  TF_RETURN_IF_ERROR(ReadBoolFromEnvVar("ENABLE_ELASTIC", 
                                        false, &enable_elastic_training));
  
  if (!enable_elastic_training) return Status::OK();

  TF_RETURN_IF_ERROR(UpdatePartitionNums());

  Graph *graph = options.graph->get();
  if (graph == nullptr)
    return errors::Internal("a graph should be available.");

  std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
  CopyGraph(*graph, new_graph.get());

  TF_RETURN_IF_ERROR(RewriteTrainingGraph(new_graph.get()));

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
  LOG(INFO) << " partition_nums_ is " << partition_nums_;
  return Status::OK();
}

Status ElasticTrainingPass::RewriteTrainingGraph(Graph* g, bool is_test) {
  std::unordered_map<std::string, Node*> ev_nodes_map;
  std::unordered_map<std::string, std::pair<bool, int>> ev_metas_map;
  std::unordered_map<std::string, std::vector<Node*>> ev_to_origin_map;
  TF_RETURN_IF_ERROR(InitEVMeta(g, ev_nodes_map, ev_metas_map, ev_to_origin_map));
  TF_RETURN_IF_ERROR(InitNewPartitionSubGraph(g, ev_metas_map, ev_nodes_map, ev_to_origin_map, is_test));
  TF_RETURN_IF_ERROR(InitNewSaveSubGraph(g, ev_metas_map, ev_nodes_map, ev_to_origin_map));

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
        std::vector<Node*> ev_vec(partition_nums_ + 20 /* hack*/, nullptr);
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
        std::vector<Node*> ev_vec(partition_nums_ + 20 /* hack*/, nullptr);
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

Status ElasticTrainingPass::InitNewSaveSubGraph(Graph* g,
                                std::unordered_map<std::string, std::pair<bool, int>>& ev_metas_map,
                                std::unordered_map<std::string, Node*>& ev_nodes_map,
                                std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map){
  std::vector<Node*> save_node_vec;
  for (auto* node: g->nodes()) {
    if (node->type_string() == "SaveV3") {
      save_node_vec.emplace_back(node);
    }
  }
  if (save_node_vec.size() == 0) return Status::OK();
  LOG(INFO) << save_node_vec.size() << " ===== " << partition_nums_;

  if (save_node_vec.size() < partition_nums_) {
    for (int i = save_node_vec.size() ; i < partition_nums_; ++i) {
      std::vector<Node*> kv_lookup_resource_node_vec;
      std::vector<string> ev_names_vec;
      std::vector<DataType> key_data_types;
      std::string assigned_device_name = "";
      Status s;
      for (auto it : ev_metas_map) {
        auto ev_node = ev_to_origin_map[it.first][i];
        if ( ev_node == nullptr) {LOG(ERROR) << it.first << " === " << i;}
        DataType key_type, value_type;
        TF_RETURN_IF_ERROR(GetNodeAttr(ev_node->attrs(), "Tkeys", &key_type));
        TF_RETURN_IF_ERROR(GetNodeAttr(ev_node->attrs(), "dtype", &value_type));
        NodeDef kv_lookup_resource_node_def;
        TF_RETURN_IF_ERROR(NodeDefBuilder(ev_node->name() + "/KvResourceLookupResource", "KvResourceLookupResource")
                                          .Input(ev_node->name(), 0, ev_node->output_type(0))
                                          .Attr("Tkeys", key_type)
                                          .Attr("dtype", value_type)
                                          .Device(ev_node->assigned_device_name())
                                          .Finalize(&kv_lookup_resource_node_def)); 
        Node* kv_lookup_resource_node = g->AddNode(kv_lookup_resource_node_def, &s);
        TF_RETURN_IF_ERROR(s);
        kv_lookup_resource_node_vec.emplace_back(kv_lookup_resource_node);
        ev_names_vec.emplace_back(ev_node->name());
        key_data_types.emplace_back(key_type);
        if (assigned_device_name == "") assigned_device_name = ev_node->assigned_device_name();
      }
      Node* node = save_node_vec[0];
      Node* save_node = g->CopyNode(node);
      save_node->set_name(node->name() + "/Copy");

      Node* sharded_filename;
      TF_RETURN_IF_ERROR(node->input_node(0, &sharded_filename));
      Node* new_sharded_filename = g->CopyNode(sharded_filename);
      new_sharded_filename->set_name(sharded_filename->name() + "/Copy");
      new_sharded_filename->set_assigned_device_name(assigned_device_name);
      g->AddEdge(new_sharded_filename, 0, save_node, 0);

      Node* prefix_name;
      TF_RETURN_IF_ERROR(sharded_filename->input_node(0, &prefix_name));
      LOG(INFO) << "STEP 1 +++++++++";
      prefix_name->ClearAttr("N");
      prefix_name->AddAttr("N", partition_nums_);
      g->AddEdge(prefix_name, 0, new_sharded_filename, 0);

      Node* num_shards;
      TF_RETURN_IF_ERROR(sharded_filename->input_node(2, &num_shards));
      LOG(INFO) << "STEP 2 +++++++++";
      num_shards->ClearAttr("value");
      num_shards->AddAttr("value", std::to_string(partition_nums_));
      g->AddEdge(num_shards, 0, new_sharded_filename, 2);

      Node* id_shards;
      TF_RETURN_IF_ERROR(sharded_filename->input_node(1, &id_shards));
      Node* new_id_shards = g->CopyNode(id_shards);
      LOG(INFO) << "STEP 3 +++++++++";
      new_id_shards->ClearAttr("value");
      new_id_shards->AddAttr("value", std::to_string(i));
      LOG(INFO) << "STEP 4 +++++++++";
      new_id_shards->ClearAttr("dtype");
      new_id_shards->AddAttr("dtype", DT_INT32);
      g->AddEdge(new_id_shards, 0, new_sharded_filename, 1);
      

      //tensor_names
      NodeDef tensor_name_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(save_node->name() + "/tensor_names", "Const")
                                        .Attr("value", "global_step")
                                        .Attr("dtype", DT_STRING)
                                        .Device(assigned_device_name)
                                        .Finalize(&tensor_name_node_def)); 
      Node* tensor_name_node = g->AddNode(tensor_name_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      g->AddEdge(tensor_name_node, 0, save_node, 1);

      //shape_and_slices
      NodeDef shape_slice_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(save_node->name() + "/shape_and_slices", "Const")
                                        .Attr("value", "")
                                        .Attr("dtype", DT_STRING)
                                        .Device(assigned_device_name)
                                        .Finalize(&shape_slice_node_def)); 
      Node* shape_slice_node = g->AddNode(shape_slice_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      g->AddEdge(shape_slice_node, 0, save_node, 2);

      //ev_names
      LOG(INFO) << "STEP 5 +++++++++";
      NodeDef ev_name_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(save_node->name() + "/ev_names", "Const")
                                        .Attr("value", ev_names_vec)
                                        .Attr("dtype", DT_STRING)
                                        .Device(assigned_device_name)
                                        .Finalize(&ev_name_node_def)); 
      Node* ev_name_node = g->AddNode(ev_name_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      g->AddEdge(ev_name_node, 0, save_node, 3);

      LOG(INFO) << "STEP 6 +++++++++";
      std::vector<NodeDefBuilder::NodeOut> kv_lookup_resource_input;
      for (auto* n: kv_lookup_resource_node_vec) {
        kv_lookup_resource_input.emplace_back(n->name(), 0, n->output_type(0));
      }      
      //ev_resources
      NodeDef kv_lookup_resource_node_def;
      int n = kv_lookup_resource_node_vec.size();
      TF_RETURN_IF_ERROR(NodeDefBuilder(save_node->name() + "/ev_resources", "Pack")
                                        .Input(kv_lookup_resource_input)
                                        .Attr("N", n)
                                        .Attr("T", key_data_types[0])
                                        .Attr("axis", 0)
                                        .Device(assigned_device_name)
                                        .Finalize(&kv_lookup_resource_node_def)); 
      Node* kv_lookup_resource_node = g->AddNode(kv_lookup_resource_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      g->AddEdge(kv_lookup_resource_node, 0, save_node, 4);
      LOG(INFO) << "STEP 7 +++++++++";
      //global_step
      Node* global_step_node;
      TF_RETURN_IF_ERROR(node->input_node(5, &global_step_node));
      Node* new_gs = g->CopyNode(global_step_node);
      new_gs->set_name(global_step_node->name() + "/Copy");
      new_gs->set_assigned_device_name(assigned_device_name);
      g->AddEdge(new_gs, 0, save_node, 5);
      LOG(INFO) << "STEP 8 +++++++++";
      for (auto* o_edge: node->out_edges()) {
        if (o_edge->IsControlEdge()) {
          Node* save_control_node = g->CopyNode(o_edge->dst());
          save_control_node->set_assigned_device_name(assigned_device_name);
          save_control_node->set_name(o_edge->dst()->name() + "/Copy");
          g->AddEdge(new_sharded_filename, 0, save_control_node, 0);

          for (auto* oo_edge: o_edge->dst()->out_edges()) {
            if (oo_edge->IsControlEdge()) {
              auto* dst_node = oo_edge->dst();
              g->AddControlEdge(save_control_node, dst_node);
              if (dst_node->type_string() == "Pack") {
                int part_num;
                TF_RETURN_IF_ERROR(GetNodeAttr(dst_node->attrs(), "N", &part_num));
                if (part_num != partition_nums_) {
                  dst_node->ClearAttr("N");
                  dst_node->AddAttr("N", partition_nums_);
                }
                g->AddEdge(new_sharded_filename, 0, dst_node, i);
              }
            }
          }
        }
      }
    }
  } else if (save_node_vec.size() > partition_nums_) {
    for (int i = partition_nums_ ; i < save_node_vec.size(); ++i) {
      Node* node = save_node_vec[i];
      Node* dst_node;
      for (auto* o_edge: node->out_edges()) {
        if (o_edge->IsControlEdge()) {
          for (auto* oo_edge: o_edge->dst()->out_edges()) {
            if (oo_edge->IsControlEdge()) {
              if (oo_edge->dst()->type_string() == "Pack") {
                dst_node = oo_edge->dst();
              }
            }
          }
        }
      }
      std::vector<Node*> nodes_to_delete;
      auto enter = [&](Node* n) {
        int device_id = n->assigned_device_name_index();
        if (device_id >= partition_nums_) {
          nodes_to_delete.emplace_back(n);
        }
        // auto device_name = n->assigned_device_name();;
        // std::string task_str = "task:";
        // auto idx_begin = device_name.rfind(task_str);
        // auto idx_end = device_name.find("device:", idx_begin);
        // int device_id = std::stoi(device_name.substr(idx_begin+task_str.size(), idx_end));
      };
      ReverseDFSFrom(*g, {dst_node}, enter, nullptr, NodeComparatorName());

      for (auto* node: nodes_to_delete) {
        g->RemoveNode(node);
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::InitNewPartitionSubGraph(Graph* g,
                                std::unordered_map<std::string, std::pair<bool, int>>& ev_metas_map,
                                std::unordered_map<std::string, Node*>& ev_nodes_map,
                                std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                bool is_test) {
  std::unordered_map<std::string, Node*> ev_to_primary_map;
  for (auto it : ev_metas_map) {
    auto ori_ev_name = it.first;
    int ev_partition_num = it.second.second;
    std::vector<Node*> nodes_to_delete;
    bool is_primary = false;
    if (ev_partition_num == partition_nums_) continue; //Do nothing

    if (ev_partition_num < partition_nums_) {
      for (int i = ev_partition_num; i < partition_nums_; ++i) {
        auto ev_node = ev_to_origin_map[ori_ev_name][0];
        std::string op_name;
        if (std::get<0>(it.second)) {
          op_name = ori_ev_name + "/" + kPart + std::to_string(i);
          is_primary = true;
        } else {
          auto sep_idx = ori_ev_name.rfind("/");
          op_name = ori_ev_name.substr(0, sep_idx) + "/" + kPart + std::to_string(i) + ori_ev_name.substr(sep_idx);
        }
        
        auto device_name = ev_node->assigned_device_name(); //std::string("/job:ps/replica:0/task:0/device:CPU:2");
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
        LOG(INFO) << "JUNQI ===>" << ori_ev_name 
                  << " === " << i
                  << " === " << op_name;
        ev_to_origin_map[ori_ev_name][i] = new_ev_node;

        // InitializeEVResource
        for (auto* o_node:  ev_node->out_nodes()) {
          if (o_node->type_string() == "InitializeKvVariableOp") {
            auto* init_node = g->CopyNode(o_node);
            init_node->set_name(o_node->name() + "/Copy");
            init_node->set_assigned_device_name(new_device_name);
            g->AddEdge(new_ev_node, 0, init_node, 0);

            //primary_ev
            if (ev_to_primary_map.count(primary_ev_name) == 0){
              LOG(ERROR) << "BUG, ev should have primary ev node";
            }
            auto* primary_node = ev_to_primary_map[primary_ev_name];
            g->AddEdge(primary_node, 0, init_node, 1);
            //init_value
            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
            auto* init_value_node = g->CopyNode(init_value_edge->src());
            init_value_node->set_name(init_value_edge->src()->name() + "/Copy");
            init_value_node->set_assigned_device_name(new_device_name);
            g->AddEdge(init_value_node, init_value_edge->src_output(), init_node, 2);
            
            //empty_key
            const Edge* empty_key_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
            auto* empty_key_node = g->CopyNode(empty_key_edge->src());
            empty_key_node->set_name(empty_key_edge->src()->name() + "/Copy");
            empty_key_node->set_assigned_device_name(new_device_name);
            g->AddEdge(empty_key_node, empty_key_edge->src_output(), init_node, 3);
          }
        }
        LOG(INFO) << "Rewriting Gather --------- ";

        // Gather
        for (auto* o_node: ev_node->out_nodes()) {
          if (o_node->type_string() == "KvResourceGather") {
            LOG(INFO) << "Step 1 --------- ";
            Node* gather_op = g->CopyNode(o_node);
            gather_op->set_name(o_node->name() + "/Copy");
            gather_op->set_assigned_device_name(new_device_name);
            g->AddEdge(new_ev_node, 0, gather_op, 0);
            const Edge* gather_id_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(1, &gather_id_edge));
            g->AddEdge(gather_id_edge->src(), gather_id_edge->src_output(), gather_op, 1);
            LOG(INFO) << "Step 2 --------- ";
            const Edge* axis_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &axis_edge));
            Node* axis = g->CopyNode(axis_edge->src());
            axis->set_name(axis_edge->src()->name() + "/Copy");
            axis->set_assigned_device_name(new_device_name);
            g->AddEdge(axis, 0, gather_op, 2);
            for (auto* o_edge: o_node->out_edges()) {
              if (o_edge->dst()->type_string() == "Identity") {
                LOG(INFO) << "Step 3 --------- ";
                Node* identity_op = g->CopyNode(o_edge->dst());
                identity_op->set_name(o_edge->dst()->name() + "/Copy");
                identity_op->set_assigned_device_name(new_device_name);
                g->AddEdge(gather_op, 0, identity_op, 0);
                LOG(INFO) << "Step 4 --------- ";
                for (auto* oo_edge: o_edge->dst()->out_edges()) {
                  if (o_edge->dst()->type_string() == "ParallelDynamicStitch") {

                  }
                }
              }
            }
            LOG(INFO) << "Step 6 --------- ";
          }
        }
      }
    } else {
      for (int i = partition_nums_; i < ev_partition_num; ++i) {
        Node* ev_node = ev_to_origin_map[ori_ev_name][i];
        nodes_to_delete.emplace_back(ev_node);

        // InitializeEVResource
        for (auto* o_node: ev_node->out_nodes()) {
          if (o_node->type_string() == "InitializeKvVariableOp") {
            nodes_to_delete.emplace_back(o_node);
            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
            nodes_to_delete.emplace_back(init_value_edge->src());
            const Edge* empty_key_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
            nodes_to_delete.emplace_back(empty_key_edge->src());
          }
        }

        // Gather
        for (auto* o_node: ev_node->out_nodes()) {
          if (o_node->type_string() == "KvResourceGather") {
            nodes_to_delete.emplace_back(o_node);
            const Edge* axis_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &axis_edge));
            nodes_to_delete.emplace_back(axis_edge->src());
            for (auto* o_edge: o_node->out_edges()) {
              if (o_edge->dst()->type_string() == "Identity") {
                nodes_to_delete.emplace_back(o_edge->dst());
              }
            }
          }
        }
      }
    }
    LOG(INFO) << "PostProcessing --------- ";
    auto ev_vec = ev_to_origin_map[ori_ev_name];
    if (!is_test) {
      if (ev_partition_num < partition_nums_) {
        TF_RETURN_IF_ERROR(ScalingUpRedistributionGraph(g, ev_vec, ev_partition_num));
        if (is_primary) {
          TF_RETURN_IF_ERROR(RewriteElasticPartitionGraph(g, ev_vec));
        }
        TF_RETURN_IF_ERROR(ScalingUpBackWardGraph(g, ev_vec, ev_partition_num));
      } else if (partition_nums_  < ev_partition_num) {
        TF_RETURN_IF_ERROR(ScalingDownRedistributionGraph(g, ev_vec, ev_partition_num));
        if (is_primary) {
          TF_RETURN_IF_ERROR(RewriteElasticPartitionGraph(g, ev_vec));
        }
        TF_RETURN_IF_ERROR(ScalingDownBackWardGraph(g, ev_vec, ev_partition_num));
      }
    }
    for (auto* node: nodes_to_delete) {
      g->RemoveNode(node);
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownBackWardGraph(Graph* g,
                                                  std::vector<Node*>& ev_node_vec,
                                                  int ev_partition_num) {
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpBackWardGraph(Graph* g,
                                                  std::vector<Node*>& ev_node_vec,
                                                  int ev_partition_num) {
  auto* ev_node = ev_node_vec[0];
  Node* ev_apply_node;
  std::vector<Node*> backward_subgraph;
  for (auto* node: ev_node->out_nodes()) {
    if (node->IsKvSparseApply()) {
      ev_apply_node = node;
    }
  }
  std::vector<Node*> nodes_to_copy;
  auto enter = [&](Node* n) {
    string device_name = n->assigned_device_name();
    if (device_name.find("ps") != string::npos) {
      nodes_to_copy.emplace_back(n);
    }
  };
  ReverseDFSFrom(*g, {ev_apply_node}, enter, nullptr, NodeComparatorName());
  
  for (int i = ev_partition_num; i < partition_nums_; ++i) {
    
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpRedistributionGraph(Graph* g,
                                                       std::vector<Node*>& ev_node_vec,
                                                       int ev_partition_num) {
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  filtered_node_vec.reserve(partition_nums_);
  for (int i = 0 ; i < partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    
    if (i < ev_partition_num) {
      for (auto* o_node: ev_node->out_nodes()) {
        if (o_node->type_string() == "FilterStorage") {
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "dtype", &value_type));
          o_node->ClearAttr("new_partition_nums");
          o_node->AddAttr("new_partition_nums", partition_nums_);
          filtered_node_vec.push_back(o_node);
        }
      }
    } else {
      NodeDef filter_storage_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(ev_node->name() + "/FilterStorage", "FilterStorage")
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
  for (int i = 0; i < partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    
    if (i < ev_partition_num) {
      for (auto* o_node: ev_node->out_nodes()) {
        if (o_node->type_string() == "ImportStorage") {
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
    TF_RETURN_IF_ERROR(NodeDefBuilder(ev_node->name() + "/ImportStorage", "ImportStorage")
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

Status ElasticTrainingPass::ScalingDownRedistributionGraph(Graph* g,
                                                       std::vector<Node*>& ev_node_vec,
                                                       int ev_partition_num) {
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  filtered_node_vec.reserve(partition_nums_);
  std::vector<Node*> delete_nodes_vec;

  for (int i = 0 ; i < partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    if (i >= partition_nums_) {
      for (auto* o_node: ev_node->out_nodes()) {
        if ((o_node->type_string() == "FilterStorage") || 
           (o_node->type_string() == "ImportStorage")) {
          delete_nodes_vec.emplace_back(o_node);
        }
      }
    } else {
      for (auto* o_node: ev_node->out_nodes()) {
        if (o_node->type_string() == "FilterStorage") {
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "dtype", &value_type));
          o_node->ClearAttr("new_partition_nums");
          o_node->AddAttr("new_partition_nums", partition_nums_);
          filtered_node_vec.push_back(o_node);
        } else if (o_node->type_string() == "ImportStorage") {
          delete_nodes_vec.emplace_back(o_node);
        }
      }
    }
  }

  for (int i = 0; i < partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    
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
    TF_RETURN_IF_ERROR(NodeDefBuilder(ev_node->name() + "/ImportStorage", "ImportStorage")
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

  for (auto* node: delete_nodes_vec) { g->RemoveNode(node);}
  return Status::OK();
}

Status ElasticTrainingPass::RewriteElasticPartitionGraph(Graph* g,
                                                         std::vector<Node*>& ev_node_vec) {
  Status s;
  Node* dynamic_partition_node = nullptr;
  Node* dynamic_stitch_node = nullptr;
  std::vector<Node*> identity_node_vec;
  std::vector<Node*> gather_node_vec;
  for (int i = 0; i < partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == "KvResourceGather") {
        gather_node_vec.push_back(o_node);
        const Edge* input_edge = nullptr;
        TF_RETURN_IF_ERROR(o_node->input_edge(1, &input_edge));
        if (input_edge->src()->type_string() == kDynamicPartition) {
          dynamic_partition_node = input_edge->src();
        }

        for (auto* oo_node: o_node->out_nodes()) {
          if (oo_node->type_string() == "Identity") {
            identity_node_vec.push_back(oo_node);
            for (auto* ooo_node: oo_node->out_nodes()) {
              if (ooo_node->type_string() == "ParallelDynamicStitch") {
                dynamic_stitch_node = ooo_node;
              }
            }
          }
        }
      }
    }
  }

  if ((dynamic_stitch_node == nullptr) || (dynamic_partition_node == nullptr)) {
    return errors::Internal("dynamic_stitch_node or dynamic_partition_node is nullptr");
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
  TF_RETURN_IF_ERROR(NodeDefBuilder(pre_node_name + "/ElasticPartition", "ElasticPartition")
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
      TF_RETURN_IF_ERROR(o_node->input_edge(0, &data_input_edge));
      if (data_input_edge->src()->type_string() != "Range") { // ID
        //Input
        g->AddEdge(data_input_edge->src(), input_edge->src_output(), elastic_node, 0);
        for (auto* o_edge: o_node->out_edges()) {
          if (o_edge->dst()->type_string() == "KvResourceGather") continue;
          g->AddEdge(elastic_node, o_edge->src_output(), o_edge->dst(), o_edge->dst_input());
        }
        delete_nodes.push_back(o_node);

      } else { // Indices
        //Input
        g->AddEdge(data_input_edge->src(), input_edge->src_output(), elastic_node, 1);
        //Output
        for (auto* o_edge: o_node->out_edges()) {
          if (o_edge->dst()->type_string() == "ParallelDynamicStitch") continue;
          g->AddEdge(elastic_node, partition_nums_ + o_edge->src_output(), o_edge->dst(), o_edge->dst_input());
        }
        delete_nodes.push_back(o_node);
      }
    }
  }
  
  Node* new_dynamic_stitch_node = g->CopyNode(dynamic_stitch_node);
  new_dynamic_stitch_node->ClearAttr("N");
  new_dynamic_stitch_node->AddAttr("N", partition_nums_);
  delete_nodes.push_back(dynamic_stitch_node);
  for (int i = 0; i < identity_node_vec.size(); ++i) {
    g->AddEdge(elastic_node, partition_nums_+i, new_dynamic_stitch_node, i);
    g->AddEdge(identity_node_vec[i], 0, new_dynamic_stitch_node, partition_nums_+i);
  }

  for (int i = 0; i < gather_node_vec.size(); ++i) {
    g->UpdateEdge(elastic_node, i, gather_node_vec[i], 1);
  }

  delete_nodes.push_back(input_edge->src());

  for (auto* n: delete_nodes) { g->RemoveNode(n); }
  return s;
}


REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0, ElasticTrainingPass);

} // namespace tensorflow