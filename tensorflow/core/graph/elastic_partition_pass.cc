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

#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/algorithm.h"

constexpr char kEnableElasticEnv[] = "ENABLE_ELASTIC";
constexpr char kDynamicPartition[] = "DynamicPartition";
constexpr char kPart[] = "part_";
constexpr char kEvInitOp[] = "InitializeKvVariableV2Op";
constexpr char kEvImportOp[] = "ImportStorage";
constexpr char kEvExportOp[] = "FilterStorage";
constexpr char kSaveOp[] = "SaveV3";

namespace tensorflow {

int ElasticTrainingPass::ori_partition_nums_ = 0;

inline string NewNodeName(const string& ori_name, int partition_id) {
  auto part_idx = ori_name.find(kPart);
  std::string pre_str = ori_name.substr(0, part_idx-1);
  std::string post_str = ori_name.substr(part_idx+strlen(kPart));
  auto post_idx = post_str.find("/");
  if (post_idx == string::npos) {
    return pre_str + "/" + kPart + std::to_string(partition_id);
  } else {
    return pre_str + "/" + kPart + std::to_string(partition_id) + post_str.substr(post_idx);
  }
}

inline Node* CopyNode(Graph* g, Node* node, 
                      const std::string& device_name,
                      const std::string& node_name) {
  Node* ret = g->CopyNode(node);
  if (node_name == "") {
    ret->set_name(node->name() + "/Copy");
  } else {
    ret->set_name(node_name);
  }
  ret->set_assigned_device_name(device_name);
  ret->ClearAttr("_class");
  return std::move(ret);
}

Status ElasticTrainingPass::Run(const GraphOptimizationPassOptions& options) {
  bool enable_elastic_training = false;
  TF_RETURN_IF_ERROR(ReadBoolFromEnvVar(kEnableElasticEnv, false,
                                        &enable_elastic_training));
  
  if (!enable_elastic_training) {
    LOG(INFO) << "Elastic training not enable.";
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(UpdatePartitionNums());
  if (ori_partition_nums_ == 0) {
    ori_partition_nums_ = partition_nums_;
    return Status::OK();
  } else if (ori_partition_nums_ == partition_nums_) {
    LOG(INFO) << "No need to redistribution";
    return Status::OK();
  } else {
    ori_partition_nums_ = partition_nums_;
  }

  Graph *graph = options.graph->get();
  if (graph == nullptr)
    return errors::Internal("a graph should be available.");
  std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
  CopyGraph(*graph, new_graph.get());


  TF_RETURN_IF_ERROR(RewriteTrainingGraph(new_graph.get()));

  DumpGraphToFile("ElasticTraining", *new_graph.get(), options.flib_def);
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
  std::unordered_map<std::string, int> primary_ev_metas_map;
  std::unordered_map<std::string, std::vector<std::string>> primary_ev_to_opt_map;
  std::unordered_map<std::string, std::vector<Node*>> ev_to_origin_map;
  TF_RETURN_IF_ERROR(InitEVMeta(g, primary_ev_metas_map, primary_ev_to_opt_map, ev_to_origin_map));
  TF_RETURN_IF_ERROR(RewriteTrainingSubGraph(g, primary_ev_metas_map, primary_ev_to_opt_map, ev_to_origin_map, is_test));
  TF_RETURN_IF_ERROR(RewriteSavingSubGraph(g, primary_ev_metas_map, primary_ev_to_opt_map, ev_to_origin_map));
  
  return Status::OK();
}

Status ElasticTrainingPass::InitEVMeta(Graph* g,
                                      std::unordered_map<std::string, int>& primary_ev_metas_map,
                                      std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map) {
  std::unordered_map<std::string, Node*> ev_nodes_map;
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
      if (primary_ev_metas_map.find(pre_str) == primary_ev_metas_map.end()) {
        primary_ev_metas_map.emplace(pre_str, 1);
        std::vector<Node*> ev_vec(partition_nums_ + 20 /* hack*/, nullptr);
        ev_vec[ev_idx] = it.second;
        ev_to_origin_map.emplace(pre_str, std::move(ev_vec));
      } else {
        primary_ev_metas_map[pre_str]++;
        ev_to_origin_map[pre_str][ev_idx] = it.second;
      }
    } else {
      int ev_idx = std::stoi(post_str.substr(0, post_idx));
      string opt_name = pre_str + post_str.substr(post_idx);
      if (primary_ev_metas_map.find(opt_name) == primary_ev_metas_map.end()) {
        primary_ev_metas_map.emplace(opt_name, 1);
        std::vector<Node*> ev_vec(partition_nums_ + 20 /* hack*/, nullptr);
        ev_vec[ev_idx] = it.second;
        ev_to_origin_map.emplace(opt_name, std::move(ev_vec));
      } else {
        primary_ev_metas_map[opt_name]++;
        ev_to_origin_map[opt_name][ev_idx] = it.second;
      }
      //exactly once
      if (ev_idx == 0) {
        auto sep_idx = opt_name.rfind("/");
        string primary_ev_name = opt_name.substr(0, sep_idx);
        LOG(INFO) << "primary string : " << pre_str
                  << " opt string: " << opt_name;
        if (primary_ev_to_opt_map.find(primary_ev_name) == primary_ev_to_opt_map.end()) {
          primary_ev_to_opt_map.emplace(primary_ev_name, std::vector<string>{opt_name});
        } else {
          primary_ev_to_opt_map[primary_ev_name].emplace_back(opt_name);
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::RewriteSavingSubGraph(Graph* g,
                                std::unordered_map<std::string, int>& primary_ev_metas_map,
                                std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map){
  /*Normaly, there is saveV3 in each PS*/
  std::vector<Node*> save_node_vec;
  for (auto* node: g->nodes()) {
    if (node->type_string() == kSaveOp) {
      save_node_vec.emplace_back(node);
    }
  }

  if (save_node_vec.size() == 0) {
    LOG(INFO) << "There is no saveV3 Op in Graph";
    return Status::OK();
  }
  

  if (save_node_vec.size() < partition_nums_) {
    for (int i = save_node_vec.size() ; i < partition_nums_; ++i) {
      std::vector<Node*> kv_lookup_resource_node_vec;
      std::vector<string> ev_names_vec;
      std::vector<DataType> key_data_types;
      std::string assigned_device_name = "";
      Status s;

      for (auto& it: primary_ev_metas_map) {
        auto ev_node = ev_to_origin_map[it.first][i];
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
      
      Node* ori_save_node = save_node_vec[0];
      Node* new_sharded_filename;
      Node* tensor_name_node;
      Node* shape_slice_node;
      Node* ev_name_node;
      Node* kv_lookup_resource_node;

      {
        Node* sharded_filename;
        TF_RETURN_IF_ERROR(ori_save_node->input_node(0, &sharded_filename));
        new_sharded_filename = CopyNode(g, sharded_filename, assigned_device_name, 
                                              sharded_filename->name() + "_" + std::to_string(i));

        Node* prefix_name;
        TF_RETURN_IF_ERROR(sharded_filename->input_node(0, &prefix_name));
        g->AddEdge(prefix_name, 0, new_sharded_filename, 0);

        Node* num_shards;
        TF_RETURN_IF_ERROR(sharded_filename->input_node(2, &num_shards));
        Tensor new_tensor_nums(DT_INT32, TensorShape({}));
        new_tensor_nums.flat<int32>()(0) = partition_nums_;
        num_shards->ClearAttr("value");
        num_shards->AddAttr("value", new_tensor_nums);
        g->AddEdge(num_shards, 0, new_sharded_filename, 2);

        Node* id_shards;
        TF_RETURN_IF_ERROR(sharded_filename->input_node(1, &id_shards));
        Node* new_id_shards = CopyNode(g, id_shards, assigned_device_name, 
                                      new_sharded_filename->name() + "/shard");
        Tensor new_tensor_ids(DT_INT32, TensorShape({}));
        new_tensor_ids.flat<int32>()(0) = i;
        new_id_shards->ClearAttr("value");
        new_id_shards->AddAttr("value", new_tensor_ids);
        g->AddEdge(new_id_shards, 0, new_sharded_filename, 1);
      }
      
      {
        //tensor_names
        Tensor new_tensor_names(DT_STRING, TensorShape({1}));
        new_tensor_names.flat<tstring>()(0) = "global_step";
        NodeDef tensor_name_node_def;
        TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" + std::to_string(i) + "/tensor_names", "Const")
                                          .Attr("value", new_tensor_names)
                                          .Attr("dtype", DT_STRING)
                                          .Device(assigned_device_name)
                                          .Finalize(&tensor_name_node_def)); 
        tensor_name_node = g->AddNode(tensor_name_node_def, &s);
        TF_RETURN_IF_ERROR(s);
      }
      
      {
        //shape_and_slices
        Tensor new_tensor_shape(DT_STRING, TensorShape({1}));
        new_tensor_shape.flat<tstring>()(0) = "";
        NodeDef shape_slice_node_def;
        TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" + std::to_string(i) + "/shape_and_slices", "Const")
                                          .Attr("value", new_tensor_shape)
                                          .Attr("dtype", DT_STRING)
                                          .Device(assigned_device_name)
                                          .Finalize(&shape_slice_node_def)); 
        shape_slice_node = g->AddNode(shape_slice_node_def, &s);
        TF_RETURN_IF_ERROR(s);
      }
      
      {
        //ev_names  
        NodeDef ev_name_node_def;
        Tensor ev_names_tensor(DT_STRING, TensorShape({ev_names_vec.size()}));
        for (int k = 0; k < ev_names_vec.size(); ++k) {
          ev_names_tensor.flat<tstring>()(k) = ev_names_vec[k];
        } 
        
        TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" + std::to_string(i)+ "/ev_names", "Const")
                                          .Attr("value", ev_names_tensor)
                                          .Attr("dtype", DT_STRING)
                                          .Device(assigned_device_name)
                                          .Finalize(&ev_name_node_def)); 
        ev_name_node = g->AddNode(ev_name_node_def, &s);
        TF_RETURN_IF_ERROR(s);
      }
      
      {
        std::vector<NodeDefBuilder::NodeOut> kv_lookup_resource_input;
        for (auto* n: kv_lookup_resource_node_vec) {
          kv_lookup_resource_input.emplace_back(n->name(), 0, n->output_type(0));
        }      

        //ev_resources
        NodeDef kv_lookup_resource_node_def;
        int n = kv_lookup_resource_node_vec.size();
        TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" + std::to_string(i) + "/ev_resources", "Pack")
                                          .Input(kv_lookup_resource_input)
                                          .Attr("N", n)
                                          .Attr("T", key_data_types[0])
                                          .Attr("axis", 0)
                                          .Device(assigned_device_name)
                                          .Finalize(&kv_lookup_resource_node_def)); 
        kv_lookup_resource_node = g->AddNode(kv_lookup_resource_node_def, &s);
        TF_RETURN_IF_ERROR(s);
      }
      
      //global_step
      Node* global_step_node;
      TF_RETURN_IF_ERROR(ori_save_node->input_node(5, &global_step_node));


      std::vector<NodeDefBuilder::NodeOut> tensors_input {NodeDefBuilder::NodeOut(global_step_node->name(), 0, global_step_node->output_type(0))};
      std::vector<DataType> ev_dtypes(kv_lookup_resource_node_vec.size(), DT_INT64);
      std::vector<DataType> n_dtypes {DT_INT64};
      //tensor_names
      NodeDef save_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" + std::to_string(i), "SaveV3")
                                        .Input(new_sharded_filename->name(), 0, new_sharded_filename->output_type(0))
                                        .Input(tensor_name_node->name(), 0, tensor_name_node->output_type(0))
                                        .Input(shape_slice_node->name(), 0, shape_slice_node->output_type(0))
                                        .Input(ev_name_node->name(), 0, ev_name_node->output_type(0))
                                        .Input(kv_lookup_resource_node->name(), 0, kv_lookup_resource_node->output_type(0))
                                        .Input(tensors_input)
                                        .Attr("dtypes", n_dtypes)
                                        .Attr("ev_key_types", ev_dtypes)
                                        .Attr("has_ev", true)
                                        .Device(assigned_device_name)
                                        .Finalize(&save_node_def));
      Node* save_node = g->AddNode(save_node_def, &s);
      TF_RETURN_IF_ERROR(s);

      for (auto* o_edge: ori_save_node->out_edges()) {
        if (o_edge->IsControlEdge()) {
          Node* save_control_node = CopyNode(g, o_edge->dst(), assigned_device_name, o_edge->dst()->name() + "_" + std::to_string(i));
          g->AddEdge(new_sharded_filename, 0, save_control_node, 0);
          g->AddControlEdge(save_node, save_control_node);
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
      Node* ori_save_node = save_node_vec[i];
      Node* dst_node;
      for (auto* o_edge: ori_save_node->out_edges()) {
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

Status ElasticTrainingPass::RewriteTrainingSubGraph(Graph* g,
                                std::unordered_map<std::string, int>& primary_ev_metas_map,
                                std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                bool is_test) {
  std::unordered_map<std::string, std::pair<Node*, Node*>> ev_to_primary_map;
  std::vector<Node*> no_op_vec(partition_nums_, nullptr);
  std::vector<Node*> init_op_vec(partition_nums_, nullptr);
  Node* import_op_main;
  for (auto* node: g->nodes()) {
    if (node->name() == "elastic_subgraph_import") {
      import_op_main = node;
    }
  }

  for (auto it : primary_ev_to_opt_map) {
    auto primary_ev_name = it.first;
    auto opt_ev_names = it.second;
    std::sort(opt_ev_names.begin(), opt_ev_names.end(), [](const std::string& str1, const std::string& str2) {
      auto part_idx = str1.rfind("/");
      std::string post_str = str1.substr(part_idx);
      auto post_idx = post_str.rfind("_");
      if (post_idx == string::npos) {
        return true;
      }
      
      auto part_idx_1 = str2.rfind("/");
      std::string post_str_1 = str2.substr(part_idx_1);
      auto post_idx_1 = post_str_1.rfind("_");
      if (post_idx_1 == string::npos) {
        return false;
      }

      return std::stoi(post_str.substr(post_idx)) < std::stoi(post_str_1.substr(post_idx_1));
    });

    int ev_partition_num = primary_ev_metas_map[primary_ev_name];
    std::vector<Node*> nodes_to_delete;
    if (ev_partition_num == partition_nums_) {
      continue; //Do nothing
    } else if (ev_partition_num < partition_nums_) {
      for (int i = ev_partition_num; i < partition_nums_; ++i) {

        auto ev_node = ev_to_origin_map[primary_ev_name][0];
        std::string op_name = primary_ev_name + "/" + kPart + std::to_string(i);
        
        auto device_name = ev_node->assigned_device_name(); //std::string("/job:ps/replica:0/task:0/device:CPU:2");
        std::string task_str = "task:";
        auto idx_begin = device_name.rfind(task_str);
        auto idx_end = device_name.find("device:", idx_begin);
        std::string new_device_name = 
            device_name.substr(0, 
                idx_begin+task_str.size()) + std::to_string(i) + device_name.substr(idx_end-1);

        Node* cur_init_op = init_op_vec[i];
        if (cur_init_op == nullptr) {
          Status s;
          NodeDef initop_def;
          TF_RETURN_IF_ERROR(NodeDefBuilder("new_sub_graph/InitOp_" +std::to_string(i), "NoOp")
                                            .Device(new_device_name)
                                            .Finalize(&initop_def));
          Node* init_node = g->AddNode(initop_def, &s);
          TF_RETURN_IF_ERROR(s);
          init_op_vec[i] = init_node;
          cur_init_op = init_node;
        }

        // EVHandler
        Node* new_ev_node = CopyNode(g, ev_node, new_device_name, op_name);
        new_ev_node->ClearAttr("shared_name");
        new_ev_node->AddAttr("shared_name", op_name);
        // new_ev_node->AddAttr("_class", "loc:@"+primary_ev_name);
        ev_to_origin_map[primary_ev_name][i] = new_ev_node;

        LOG(INFO) << "JUNQI ===>" << primary_ev_name 
                  << " === " << i;
        
        Node* primary_init_node;
        // InitializeEVResource
        for (auto* o_node: ev_node->out_nodes()) {
          if (o_node->type_string() == kEvInitOp) {
            const Node* tmp_check_ev_0;
            TF_RETURN_IF_ERROR(o_node->input_node(0, &tmp_check_ev_0));
            const Node* tmp_check_ev_1;
            TF_RETURN_IF_ERROR(o_node->input_node(1, &tmp_check_ev_1));
            if (tmp_check_ev_0->name() != tmp_check_ev_1->name()) continue;

            primary_init_node = CopyNode(g, o_node, new_device_name,
                                       new_ev_node->name() + "/InitializeKvVariableV2Op");
            g->AddEdge(new_ev_node, 0, primary_init_node, 0);
            g->AddEdge(new_ev_node, 0, primary_init_node, 1);
            //init_value
            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
            auto* init_value_node = CopyNode(g, init_value_edge->src(), new_device_name,
                                             new_ev_node->name() + "/" + init_value_edge->src()->name());
            g->AddEdge(init_value_node, init_value_edge->src_output(), primary_init_node, 2);
            
            //empty_key
            const Edge* empty_key_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
            auto* empty_key_node = CopyNode(g, empty_key_edge->src(), 
                                            new_device_name,
                                            new_ev_node->name() + "/" + empty_key_edge->src()->name());
            g->AddEdge(empty_key_node, empty_key_edge->src_output(), primary_init_node, 3);
            break;
          }
        }
        g->AddControlEdge(primary_init_node, cur_init_op);

        LOG(INFO) << "Rewriting Gather --------- ";

        // Gather
        for (auto* o_node: ev_node->out_nodes()) {
          if (o_node->type_string() == "KvResourceGather") {
            Node* gather_op = CopyNode(g, o_node, new_device_name,
                                       o_node->name() + "/Copy");
            g->AddEdge(new_ev_node, 0, gather_op, 0);
            const Edge* gather_id_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(1, &gather_id_edge));
            g->AddEdge(gather_id_edge->src(), gather_id_edge->src_output(), gather_op, 1);
            const Edge* axis_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(2, &axis_edge));
            Node* axis = CopyNode(g, axis_edge->src(), new_device_name,
                                  new_ev_node->name() + "/" + axis_edge->src()->name());
            g->AddEdge(axis, 0, gather_op, 2);
            for (auto* o_edge: o_node->out_edges()) {
              if (o_edge->dst()->type_string() == "Identity") {
                Node* identity_op = CopyNode(g, o_edge->dst(),
                                             new_device_name, "");
                g->AddEdge(gather_op, 0, identity_op, 0);
              }
            }
          }
        }

        // OptEV
        for (auto& opt_ev_name: opt_ev_names) {
          auto ev_node = ev_to_origin_map[opt_ev_name][0];
          auto sep_idx = opt_ev_name.rfind("/");
          std::string op_name = opt_ev_name.substr(0, sep_idx) +
                                     "/" + kPart + std::to_string(i) + opt_ev_name.substr(sep_idx);

          // EVHandler
          Node* new_opt_ev_node = CopyNode(g, ev_node, new_device_name, op_name);
          new_opt_ev_node->ClearAttr("shared_name");
          new_opt_ev_node->AddAttr("shared_name", op_name);
          // new_ev_node->AddAttr("_class", "loc:@"+primary_ev_name);
          
          LOG(INFO) << "JUNQI  BACKWARD ===>" << opt_ev_name 
                    << " === " << i;
          ev_to_origin_map[opt_ev_name][i] = new_opt_ev_node;

          // InitializeEVResource
          for (auto* o_node: ev_node->out_nodes()) {
            if (o_node->type_string() == kEvInitOp) {
              Node* init_node = CopyNode(g, o_node, new_device_name,
                                            new_opt_ev_node->name() + "/" + o_node->name());
              g->AddEdge(new_opt_ev_node, 0, init_node, 0);

              g->AddEdge(new_ev_node, 0, init_node, 1);
              g->AddControlEdge(primary_init_node, init_node);
              g->AddControlEdge(init_node, cur_init_op);
              //init_value
              const Edge* init_value_edge = nullptr;
              TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
              auto* init_value_node = CopyNode(g, init_value_edge->src(),
                                              new_device_name, new_opt_ev_node->name() + "/" + init_value_edge->src()->name());
              g->AddEdge(init_value_node, init_value_edge->src_output(), init_node, 2);
              
              //empty_key
              const Edge* empty_key_edge = nullptr;
              TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
              Node* empty_key_node = CopyNode(g, empty_key_edge->src(), new_device_name, "");

              g->AddEdge(empty_key_node, empty_key_edge->src_output(), init_node, 3);
            }
          }
        }
      }
    } else {
      for (int i = partition_nums_; i < ev_partition_num; ++i) {
        Node* ev_node = ev_to_origin_map[primary_ev_name][i];
        nodes_to_delete.emplace_back(ev_node);

        // InitializeEVResource
        for (auto* o_node: ev_node->out_nodes()) {
          if (o_node->type_string() == kEvInitOp) {
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

        for (auto& opt_ev_name: opt_ev_names) {
          Node* ev_node = ev_to_origin_map[opt_ev_name][i];
          nodes_to_delete.emplace_back(ev_node);

          // InitializeEVResource
          for (auto* o_node: ev_node->out_nodes()) {
            if (o_node->type_string() == kEvInitOp) {
              nodes_to_delete.emplace_back(o_node);
              const Edge* init_value_edge = nullptr;
              TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
              nodes_to_delete.emplace_back(init_value_edge->src());
              const Edge* empty_key_edge = nullptr;
              TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
              nodes_to_delete.emplace_back(empty_key_edge->src());
            }
          }
        }
      }
    }

    LOG(INFO) << "PostProcessing --------- ";

    if (!is_test) {
      if (ev_partition_num < partition_nums_) {
        std::vector<Node*> primary_ev_filters(partition_nums_, nullptr);
        TF_RETURN_IF_ERROR(ScalingUpRedistributionGraph(g, ev_to_origin_map[primary_ev_name], import_op_main, ev_partition_num, primary_ev_filters));
        for (auto& opt_ev_name: opt_ev_names) {
          TF_RETURN_IF_ERROR(ScalingUpRedistributionGraph(g, ev_to_origin_map[opt_ev_name], import_op_main, ev_partition_num, primary_ev_filters));
        }
      } else if (partition_nums_  < ev_partition_num) {
        TF_RETURN_IF_ERROR(ScalingDownRedistributionGraph(g, ev_to_origin_map[primary_ev_name], ev_partition_num));
        for (auto& opt_ev_name: opt_ev_names) {
          TF_RETURN_IF_ERROR(ScalingDownRedistributionGraph(g, ev_to_origin_map[opt_ev_name], ev_partition_num));
        }
      }
      Node* elastic_node;
      Node* p_dynamic_stitch_node;
      TF_RETURN_IF_ERROR(RewriteElasticPartitionGraph(g, ev_to_origin_map[primary_ev_name], &elastic_node, &p_dynamic_stitch_node));
      TF_RETURN_IF_ERROR(ScalingUpBackWardGraph(g, ev_to_origin_map, primary_ev_name, opt_ev_names, elastic_node, p_dynamic_stitch_node,
                                                no_op_vec, ev_partition_num));
    }

    for (auto* node: nodes_to_delete) {
      g->RemoveNode(node);
    }
  }

  
  for (auto* node: g->nodes()) {
    if (node->name() == "elastic_subgraph_init") {
      for (auto* ps_init_node: init_op_vec) {
        if (ps_init_node != nullptr) {
          g->AddControlEdge(ps_init_node, node);
        }
      }
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpBackWardGraph(Graph* g,
                                                   std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                                   const std::string& primary_ev_name,
                                                   const std::vector<std::string>& opt_ev_names,
                                                   Node* elastic_node, Node* p_dynamic_stitch_node,
                                                   std::vector<Node*>& no_op_vec,
                                                   int ev_partition_num) {
  LOG(INFO) << "Backward Pass";
  Node* ev_node = ev_to_origin_map[primary_ev_name][0];
  Node* new_reshape_1;
  for (int i = ev_partition_num; i < partition_nums_; ++i) {
    Node* cur_ev_node = ev_to_origin_map[primary_ev_name][i];
    Node* cur_noop_node = no_op_vec[i];
    string new_device_name = cur_ev_node->assigned_device_name();
    for (auto* node: ev_node->out_nodes()) {
      if (node->IsKvSparseApply()) {
        if (cur_noop_node == nullptr) {
          Status s;
          NodeDef noop_def;
          TF_RETURN_IF_ERROR(NodeDefBuilder("head/Optimizer/update/NoOp_" +std::to_string(i), "NoOp")
                                            .Device(new_device_name)
                                            .Finalize(&noop_def));
          Node* no_node = g->AddNode(noop_def, &s);
          TF_RETURN_IF_ERROR(s);
          for (auto* edge: node->out_edges()) {
            for (auto* o_edge: edge->dst()->out_edges()) {
              if (o_edge->IsControlEdge()) {
                g->AddControlEdge(no_node, o_edge->dst());
              }
            }
          }
          cur_noop_node = no_node;
          no_op_vec[i] = no_node;
        }

        LOG(INFO) << "Copying new apply node";
        Node* new_apply_node = CopyNode(g, node, new_device_name, NewNodeName(node->name(), i));
        g->AddControlEdge(new_apply_node, cur_noop_node);
        g->AddEdge(ev_to_origin_map[primary_ev_name][i], 0, new_apply_node, 0);
        for (int j = 0; j < opt_ev_names.size(); ++j) {
          g->AddEdge(ev_to_origin_map[opt_ev_names[j]][i], 0, new_apply_node, j+1);
        }
        Node* new_unique;
        Node* new_expand_dims;
        for (int j = node->num_inputs() - 1;j > opt_ev_names.size() ;--j) {
          Node* i_node;
          TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
          if (i_node->type_string() == "Unique") {
            LOG(INFO) << "Copying new unique node";
            new_unique = CopyNode(g, i_node, new_device_name,
                                  NewNodeName(cur_ev_node->name()+"/Unique", i));
            g->AddEdge(new_unique, 0, new_apply_node, j);
            //unique INPUT 0
            LOG(INFO) << "Copying reshape of unique input";
            Node* reshape_id;
            TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_id));
            Node* new_reshape_id = CopyNode(g, reshape_id, reshape_id->assigned_device_name(),
                                            reshape_id->name() + "/Copy_" + std::to_string(i));
            g->AddEdge(new_reshape_id, 0, new_unique, 0);

            LOG(INFO) << "Copying RecordSparseIndices";
            for (auto * o_node: reshape_id->out_nodes()) {
              if (o_node->type_string() == "RecordSparseIndices") {
                Node* new_record_sparse = CopyNode(g, o_node, new_device_name,
                                                   NewNodeName(o_node->name(), i));
                g->AddEdge(new_reshape_id, 0, new_record_sparse, 0);
                g->AddControlEdge(new_record_sparse, cur_noop_node);
              }
            }

            //Reshape INPUT
            g->AddEdge(elastic_node, i, new_reshape_id, 0);

            Node* expand_dims;
            TF_RETURN_IF_ERROR(reshape_id->input_node(1, &expand_dims));
            new_expand_dims = CopyNode(g, expand_dims, expand_dims->assigned_device_name(),
                                          expand_dims->name() + "_" + std::to_string(i));
            g->AddEdge(new_expand_dims, 0, new_reshape_id, 1);

            //expand dims INPUT
            Node* expand_dims_size;
            TF_RETURN_IF_ERROR(expand_dims->input_node(0, &expand_dims_size));
            Node* new_expand_dims_size = CopyNode(g, expand_dims_size, expand_dims_size->assigned_device_name(),
                                          expand_dims_size->name() + "_" + std::to_string(i));
            g->AddEdge(new_expand_dims_size, 0, new_expand_dims, 0);
            g->AddEdge(elastic_node, i, new_expand_dims_size, 0);

            Node* expand_dims_dim;
            TF_RETURN_IF_ERROR(expand_dims->input_node(1, &expand_dims_dim));
            Node* new_expand_dims_dim = CopyNode(g, expand_dims_dim, expand_dims_dim->assigned_device_name(),
                                          expand_dims_dim->name() + "_" + std::to_string(i));
            g->AddEdge(new_expand_dims_dim, 0, new_expand_dims, 1);

          } else if (i_node->type_string() == "UnsortedSegmentSum") {
            /*
              control_dependency          Reshape ->
              ExpandDims                         
              ElasticPartition: ID -> Unique: idx ->   UnsortedSegmentSum
                                      strided_slice ->
                        |
                        v
                        SparseRecordIndices
            */
            LOG(INFO) << "Copying new UnsortedSegmentSum node";
            Node* new_unsorted_segment = CopyNode(g, i_node, new_device_name,
                                                  NewNodeName(i_node->name(), i));
            g->AddEdge(new_unsorted_segment, 0, new_apply_node, j);
            //Input 0
            {
              Node* reshape;
              TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape));
              Node* new_reshape = CopyNode(g, reshape, reshape->assigned_device_name(),
                                            reshape->name() + "_" + std::to_string(i));
              g->AddEdge(new_reshape, 0, new_unsorted_segment, 0);
              // Reshape INPUT 0
              Node* control_denpency;
              TF_RETURN_IF_ERROR(reshape->input_node(0, &control_denpency));
              Node* new_control_denpency = CopyNode(g, control_denpency, control_denpency->assigned_device_name(),
                                            control_denpency->name() + "_" + std::to_string(i));
              g->AddEdge(new_control_denpency, 0, new_reshape, 0);

              for (auto* i_edge: control_denpency->in_edges()) {
                if (i_edge->IsControlEdge()) {
                  g->AddControlEdge(i_edge->src(), new_control_denpency);
                }
              }

              //control_dependency INPUT 0
              Node* gather_1;
              TF_RETURN_IF_ERROR(control_denpency->input_node(0, &gather_1));
              Node* new_gather_1 = CopyNode(g, gather_1, gather_1->assigned_device_name(),
                                            gather_1->name() + "_" + std::to_string(i));
              g->AddEdge(new_gather_1, 0, new_control_denpency, 0);
              for (auto* o_edge: gather_1->out_edges()) {
                if (o_edge->IsControlEdge()) {
                  g->AddControlEdge(new_gather_1, o_edge->dst());
                }
              }

              Node* reshape_1;
              TF_RETURN_IF_ERROR(gather_1->input_node(0, &reshape_1));
              g->AddEdge(reshape_1, 0, new_gather_1, 0);

              //gather_1 INPUT1
              g->AddEdge(elastic_node, partition_nums_+i/*idx*/, new_gather_1, 1);
              //gather_1 INPUT2
              Node* axis_1;
              TF_RETURN_IF_ERROR(gather_1->input_node(2, &axis_1));
              Node* new_axis_1 = CopyNode(g, axis_1, axis_1->assigned_device_name(),
                                            axis_1->name() + "_" + std::to_string(i));
              g->AddEdge(new_axis_1, 0, new_gather_1, 2);

              // Reshape INPUT 1
              Node* concat;
              TF_RETURN_IF_ERROR(reshape->input_node(1, &concat));
              Node* new_concat = CopyNode(g, concat, concat->assigned_device_name(),
                                            concat->name() + "_" + std::to_string(i));
              g->AddEdge(new_concat, 0, new_reshape, 1);

              // concat INPUT 0
              g->AddEdge(new_expand_dims, 0, new_concat, 0);

              // concat INPUT 1
              Node* strided_slice;
              TF_RETURN_IF_ERROR(concat->input_node(1, &strided_slice));
              Node* new_strided_slice = CopyNode(g, strided_slice, strided_slice->assigned_device_name(),
                                            strided_slice->name() + "_" + std::to_string(i));
              g->AddEdge(new_strided_slice, 0, new_concat, 1);

              for (int k = 0; k < strided_slice->num_inputs();++k) {
                Node* partial_strided_slice;
                TF_RETURN_IF_ERROR(strided_slice->input_node(k, &partial_strided_slice));
                Node* new_node = CopyNode(g, partial_strided_slice, partial_strided_slice->assigned_device_name(),
                                              partial_strided_slice->name() + "/Copy_" + std::to_string(i));
                g->AddEdge(new_node, 0, new_strided_slice, k);
              }

              // concat INPUT 2
              Node* axis;
              TF_RETURN_IF_ERROR(concat->input_node(2, &axis));
              Node* new_axis = CopyNode(g, axis, axis->assigned_device_name(),
                                            axis->name() + "_" + std::to_string(i));
              g->AddEdge(new_axis, 0, new_concat, 2);
            }

            // Input 1
            g->AddEdge(new_unique, 1/*idx*/, new_unsorted_segment, 1);
            LOG(INFO) << "Copying  UnsortedSegmentSum node INPUT 2";
            // Input 2
            {
              Node* strided_slice;
              TF_RETURN_IF_ERROR(i_node->input_node(2, &strided_slice));
              Node* new_strided_slice = CopyNode(g, strided_slice, new_device_name,
                                                 NewNodeName(strided_slice->name(), i));
              g->AddEdge(new_strided_slice, 0, new_unsorted_segment, 2);

              Node* shape;
              TF_RETURN_IF_ERROR(strided_slice->input_node(0, &shape));
              Node* new_shape = CopyNode(g, shape, new_device_name,
                                         NewNodeName(shape->name(), i));
              g->AddEdge(new_unique, 0, new_shape, 0);
              g->AddEdge(new_shape, 0, new_strided_slice, 0);
              
              for (int k = 1; k < strided_slice->num_inputs();++k) {
                Node* partial_strided_slice;
                TF_RETURN_IF_ERROR(strided_slice->input_node(k, &partial_strided_slice));
                Node* new_node = CopyNode(g, partial_strided_slice, new_device_name,
                                          NewNodeName(partial_strided_slice->name(), i));
                g->AddEdge(new_node, 0, new_strided_slice, k);
              }
            }
          } else {
            g->AddEdge(i_node, 0, new_apply_node, j);
          }
        }
        // LOG(INFO) << "filter op op_def : " << new_apply_node->DebugString();
      }
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpRedistributionGraph(Graph* g,
                                                         std::vector<Node*>& ev_node_vec,
                                                         Node* import_op_main,
                                                         int ev_partition_num,
                                                         std::vector<Node*>& primary_ev_filters) {
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  filtered_node_vec.reserve(partition_nums_);
  for (int i = 0 ; i < partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    auto* primary_ev_filter_node = primary_ev_filters[i];
    
    if (i < ev_partition_num) {
      for (auto* o_node: ev_node->out_nodes()) {
        if (o_node->type_string() == kEvExportOp) {
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "dtype", &value_type));
          o_node->ClearAttr("new_partition_nums");
          o_node->AddAttr("new_partition_nums", partition_nums_);
          filtered_node_vec.push_back(o_node);
          if (primary_ev_filter_node == nullptr) {
            primary_ev_filters[i] = o_node;
          } else {
            g->AddControlEdge(o_node, primary_ev_filters[i]);
          }
        }
      }
    } else {
      NodeDef filter_storage_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(ev_node->name() + "/FilterStorage", kEvExportOp)
                                        .Input(ev_node->name(), 0, ev_node->output_type(0))
                                        .Attr("partition_id", i)
                                        .Attr("new_partition_nums", partition_nums_)
                                        .Attr("Tkeys", key_type)
                                        .Attr("dtype", value_type)
                                        .Device(ev_node->assigned_device_name())
                                        .Finalize(&filter_storage_node_def)); 
      Node* filter_node = g->AddNode(filter_storage_node_def, &s);
      TF_RETURN_IF_ERROR(s);

      filtered_node_vec.push_back(filter_node);
      if (primary_ev_filter_node == nullptr) {
        primary_ev_filters[i] = filter_node;
      } else {
        g->AddControlEdge(filter_node, primary_ev_filters[i]);
      }
    }
  }

  for (int i = 0; i < partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    if (i < ev_partition_num) {
      for (auto* o_node: ev_node->out_nodes()) {
        if (o_node->type_string() == kEvImportOp) {
          o_node->ClearAttr("partition_nums");
          o_node->AddAttr("partition_nums", partition_nums_-1);
          std::vector<const Edge*> in_edges;
          in_edges.reserve(o_node->in_edges().size());
          for (auto* o_edge: o_node->in_edges()) {
            in_edges.emplace_back(o_edge);
          }
          for (const Edge* e : in_edges) {
            g->RemoveEdge(e);
          }
          g->AddEdge(ev_node, 0, o_node, 0);
          int k = 0;
          for (int j = 0; j < filtered_node_vec.size(); ++j) {
            if (j != i) {
              g->AddEdge(filtered_node_vec[j], 0, o_node, 1+k);
              g->AddEdge(filtered_node_vec[j], 1, o_node, 1+(partition_nums_-1)+k);
              g->AddEdge(filtered_node_vec[j], 2, o_node, 1+(partition_nums_-1)*2+k);
              g->AddEdge(filtered_node_vec[j], 3, o_node, 1+(partition_nums_-1)*3+k);
              ++k;
            }
          }
        }
      }
    } else {
      std::vector<NodeDefBuilder::NodeOut> import_keys;
      std::vector<NodeDefBuilder::NodeOut> import_values;
      std::vector<NodeDefBuilder::NodeOut> import_versions;
      std::vector<NodeDefBuilder::NodeOut> import_freqs;
      string import_op_name = ev_node->name() + "/" + kEvImportOp;
      for (int j = 0; j < filtered_node_vec.size(); ++j) {
        if (j != i) {
          import_keys.emplace_back(filtered_node_vec[j]->name(), 0, filtered_node_vec[j]->output_type(0));
          import_values.emplace_back(filtered_node_vec[j]->name(), 1, filtered_node_vec[j]->output_type(1));
          import_versions.emplace_back(filtered_node_vec[j]->name(), 2, filtered_node_vec[j]->output_type(2));
          import_freqs.emplace_back(filtered_node_vec[j]->name(), 3, filtered_node_vec[j]->output_type(3));
        }
      }
      NodeDef import_storage_node_def;
      TF_RETURN_IF_ERROR(NodeDefBuilder(import_op_name, kEvImportOp)
                                        .Input(ev_node->name(), 0, ev_node->output_type(0))
                                        .Input(import_keys)
                                        .Input(import_values)
                                        .Input(import_versions)
                                        .Input(import_freqs)
                                        .Attr("partition_id", i)
                                        .Attr("partition_nums", partition_nums_-1)
                                        .Attr("Tkeys", key_type)
                                        .Attr("dtype", value_type)
                                        .Device(ev_node->assigned_device_name())
                                        .Finalize(&import_storage_node_def)); 
      Node* import_node = g->AddNode(import_storage_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      g->AddControlEdge(import_node, import_op_main);
      // LOG(INFO) << "filter op op_def : " << import_node->def().DebugString();
      for (int k = 0; k < ev_partition_num; ++k) {
        auto* tmp_ev_node = ev_node_vec[k];
        for (auto* n: tmp_ev_node->out_nodes()) {
          if (n->type_string() == kEvImportOp) {
            g->AddControlEdge(import_node, n);
          }
        }
      }
    }
  }

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
        if ((o_node->type_string() == kEvExportOp) || 
           (o_node->type_string() == kEvImportOp)) {
          delete_nodes_vec.emplace_back(o_node);
        }
      }
    } else {
      for (auto* o_node: ev_node->out_nodes()) {
        if (o_node->type_string() == kEvExportOp) {
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
          TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "dtype", &value_type));
          o_node->ClearAttr("new_partition_nums");
          o_node->AddAttr("new_partition_nums", partition_nums_);
          filtered_node_vec.push_back(o_node);
        } else if (o_node->type_string() == kEvImportOp) {
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
    TF_RETURN_IF_ERROR(NodeDefBuilder(ev_node->name() + "/ImportStorage", kEvImportOp)
                                      .Input(ev_node->name(), 0, ev_node->output_type(0))
                                      .Input(import_keys)
                                      .Input(import_values)
                                      .Input(import_versions)
                                      .Input(import_freqs)
                                      .Attr("partition_id", i)
                                      .Attr("partition_nums", partition_nums_-1)
                                      .Attr("Tkeys", key_type)
                                      .Attr("dtype", value_type)
                                      .Device(ev_node->assigned_device_name())
                                      .Finalize(&import_storage_node_def)); 
    Node* import_node = g->AddNode(import_storage_node_def, &s);
    TF_RETURN_IF_ERROR(s);
  }

  for (auto* node: delete_nodes_vec) { g->RemoveNode(node);}
  return Status::OK();
}

Status ElasticTrainingPass::RewriteElasticPartitionGraph(Graph* g,
                                                         std::vector<Node*>& ev_node_vec,
                                                         Node** elastic_node,
                                                         Node** p_dynamic_stitch_node) {
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
                                    .Device(dynamic_partition_node->assigned_device_name())
                                    .Finalize(&elastic_node_def)); 
  
  *elastic_node = g->AddNode(elastic_node_def, &s);
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
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(), *elastic_node, 0);
        for (auto* o_edge: o_node->out_edges()) {
          if (o_edge->dst()->type_string() == "KvResourceGather") continue;
          g->AddEdge(*elastic_node, o_edge->src_output(), o_edge->dst(), o_edge->dst_input());
        }
        delete_nodes.push_back(o_node);

      } else { // Indices
        //Input
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(), *elastic_node, 1);
        //Output
        for (auto* o_edge: o_node->out_edges()) {
          if (o_edge->dst()->type_string() == "ParallelDynamicStitch") continue;
          g->AddEdge(*elastic_node, partition_nums_ + o_edge->src_output(), o_edge->dst(), o_edge->dst_input());
        }
        delete_nodes.push_back(o_node);
      }
    }
  }
  
  *p_dynamic_stitch_node = CopyNode(g, dynamic_stitch_node,
                                    dynamic_partition_node->assigned_device_name(),
                                    dynamic_stitch_node->name());
  (*p_dynamic_stitch_node)->ClearAttr("N");
  (*p_dynamic_stitch_node)->AddAttr("N", partition_nums_);
  delete_nodes.push_back(dynamic_stitch_node);
  for (int i = 0; i < identity_node_vec.size(); ++i) {
    g->AddEdge(*elastic_node, partition_nums_+i, *p_dynamic_stitch_node, i);
    g->AddEdge(identity_node_vec[i], 0, *p_dynamic_stitch_node, partition_nums_+i);
  }

  for (int i = 0; i < gather_node_vec.size(); ++i) {
    g->UpdateEdge(*elastic_node, i, gather_node_vec[i], 1);
  }

  delete_nodes.push_back(input_edge->src());

  for (auto* n: delete_nodes) { g->RemoveNode(n); }
  return s;
}


REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0, ElasticTrainingPass);

} // namespace tensorflow