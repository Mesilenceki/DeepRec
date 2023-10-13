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

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/algorithm.h"

constexpr char kPart[] = "part_";
constexpr char kEnableElasticEnv[] = "ENABLE_ELASTIC";
constexpr char kDynamicPartition[] = "DynamicPartition";
constexpr char kParaDynamicStitch[] = "ParallelDynamicStitch";
constexpr char kEvInitOp[] = "InitializeKvVariableV2Op";
constexpr char kEvImportOp[] = "ImportStorage";
constexpr char kEvExportOp[] = "FilterStorage";
constexpr char kVariableOp[] = "VariableV2";
constexpr char kIdentityOp[] = "Identity";
constexpr char kSaveOp[] = "SaveV3";
constexpr char kRestoreOp[] = "RestoreV2";
constexpr char kElasticImportScope[] = "elastic_import";

namespace tensorflow {

int ElasticTrainingPass::cur_partition_nums_ = 0;

inline string NewDeviceName(Node* node, int i) {
  auto device_name =
      node->assigned_device_name();  
  std::string task_str = "task:";
  auto idx_begin = device_name.rfind(task_str);
  auto idx_end = device_name.find("device:", idx_begin);
  std::string new_device_name =
      device_name.substr(0, idx_begin + task_str.size()) +
      std::to_string(i) + device_name.substr(idx_end - 1);
  return new_device_name;
}

Status FindNode(Node* src, const string& target, std::function<Status(Node* target_node)> fn) {
 for (auto* o_node: src->out_nodes()) {
   if (o_node->type_string() == target) {
    fn(o_node);
   }
 }
 return Status::OK();
}

inline string GatherName(VarType var_type, Node* node) {

}

inline bool IsApplyNode(VarType var_type, Node* node) {
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      return node->IsKvSparseApply();
      break;
    case VarType::REF_VAR:
    case VarType::RESOURCE_VAR:
      return node->IsSparseApplyAdagradOps() || node->IsSparseApplyFtrlOps() || node->IsApplySparseAdamOps();
      break;
      return false;
  }
}

inline string NewNodeName(const string& ori_name, int partition_id) {
  auto part_idx = ori_name.find(kPart);
  if (part_idx == -1) {
    return ori_name + "/Copy_" + std::to_string(partition_id);
  } else {
    std::string pre_str = ori_name.substr(0, part_idx-1);
    std::string post_str = ori_name.substr(part_idx+strlen(kPart));
    auto post_idx = post_str.find("/");
    if (post_idx == string::npos) {
      return pre_str + "/" + kPart + std::to_string(partition_id);
    } else {
      return pre_str + "/" + kPart + std::to_string(partition_id) + post_str.substr(post_idx);
    }
  }
}

inline Node* CopyNode(Graph* g, Node* node, 
                      const std::string& device_name,
                      int partition_id, const string& op_name = "") {
  Node* ret = g->CopyNode(node);
  if (op_name == ""){
    ret->set_name(NewNodeName(node->name(), partition_id));
  } else {
    ret->set_name(op_name);
  }
  ret->set_assigned_device_name(device_name);
  ret->ClearAttr("_class"); // remove pre-exist colocation
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

  int partition_nums;
  TF_RETURN_IF_ERROR(UpdatePartitionNums(partition_nums));
  if (cur_partition_nums_ == 0) {
    cur_partition_nums_ = partition_nums;
    return Status::OK();
  } else if (cur_partition_nums_ == partition_nums) {
    LOG(INFO) << "No need to do elastic partition pass.";
    return Status::OK();
  } else {
    scaling_up_ = true ? partition_nums > cur_partition_nums_ : partition_nums < cur_partition_nums_;
    cur_partition_nums_ = partition_nums;
  }

  Graph *graph = options.graph->get();
  if (graph == nullptr)
    return errors::Internal("a graph should be available.");
  std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
  CopyGraph(*graph, new_graph.get());


  TF_RETURN_IF_ERROR(RewriteSubGraph(new_graph.get()));

  DumpGraphToFile("ElasticTraining", *new_graph.get(), options.flib_def);
  options.graph->swap(new_graph);
  return Status::OK();
}

Status ElasticTrainingPass::RewriteSubGraph(Graph* g, bool is_test) {
  std::unordered_map<std::string, PartitionVarMeta> primary_node_metas_map;
  std::unordered_map<std::string, std::vector<std::string>> primary_node_to_opt_map;
  std::unordered_map<std::string, std::vector<Node*>> node_to_origin_map;
  std::unordered_map<std::string, Node*> unpartitioned_node_map;
  ElasticHookMetaNode meta_node(cur_partition_nums_);
  TF_RETURN_IF_ERROR(InitHookMetaNode(g, meta_node));
  TF_RETURN_IF_ERROR(InitVarMeta(g, primary_node_metas_map, primary_node_to_opt_map, node_to_origin_map, unpartitioned_node_map));
  TF_RETURN_IF_ERROR(RewriteTrainingSubGraph(g, primary_node_metas_map, primary_node_to_opt_map, node_to_origin_map, meta_node, is_test));
  TF_RETURN_IF_ERROR(RewriteSavingSubGraph(g, primary_node_metas_map, primary_node_to_opt_map, node_to_origin_map, unpartitioned_node_map, meta_node));
  return Status::OK();
}

Status ElasticTrainingPass::MoveUnPartitionedVariable(Graph* g, std::unordered_map<std::string, Node*>& unpartitioned_node_map,
                                                      std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
                                                      std::unordered_set<Node*>& eval_nodes_to_add,
                                                      ElasticHookMetaNode& meta_node) {

  for (auto& it: unpartitioned_node_map) {
    auto set_it = nodes_to_add.find(it.second);
    auto eval_it = eval_nodes_to_add.find(it.second);
    if (set_it == nodes_to_add.end() && eval_it != eval_nodes_to_add.end()) {
      LOG(INFO) << it.second->name() << " ---- device_name";
      for (auto* o_node: it.second->out_nodes()) {
        if ((o_node->type_string() == "Identity") &&
            (o_node->name().find(kElasticImportScope) != string::npos)) {
          for (auto* oo_node: o_node->out_nodes()) {
            if (oo_node->type_string() == "Assign") {
              g->RemoveNode(oo_node);
            }
          }
        }
      }
    } else {
      for (auto* o_node: it.second->out_nodes()) {
        if ((o_node->type_string() == "Identity") &&
            (o_node->name().find(kElasticImportScope) != string::npos)) {
          for (auto* oo_node: o_node->out_nodes()) {
            if (oo_node->type_string() == "Assign") {
              Node* tmp_value;
              TF_RETURN_IF_ERROR(oo_node->input_node(0, &tmp_value));
              meta_node.m_tmp_value_init_op->set_assigned_device_name(oo_node->assigned_device_name());
              it.second->set_assigned_device_name(oo_node->assigned_device_name());
              TF_RETURN_IF_ERROR(g->UpdateEdge(tmp_value, 0, o_node, 0));
              TF_RETURN_IF_ERROR(g->UpdateEdge(it.second, 0, oo_node, 0));
              TF_RETURN_IF_ERROR(g->UpdateEdge(o_node, 0, oo_node, 1));
              g->AddControlEdge(oo_node, meta_node.m_tmp_value_init_op);
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::RewriteSavingSubGraph(
    Graph* g, std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
    std::unordered_map<std::string, std::vector<std::string>>&
        primary_node_to_opt_map,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map,
    ElasticHookMetaNode& meta_node) {
  std::vector<Node*> save_node_vec;
  std::vector<Node*> restore_node_vec;
  for (auto* node : g->nodes()) {
    if (node->type_string() == kSaveOp) {
      save_node_vec.emplace_back(node);
    } else if(node->type_string() == kRestoreOp) {
      restore_node_vec.emplace_back(node);
    }
  }

  if ((save_node_vec.size() == 0) || (restore_node_vec.size() == 0)) {
    LOG(INFO) << "There is no SaveV3 and RestoreV2 Op in Graph, Nothing to do.";
    return Status::OK();
  }

  std::unordered_set<Node*> nodes_to_delete;
  std::unordered_map<string, std::vector<int64>> variable_shape;
  std::unordered_map<Node*, std::pair<string, string>> nodes_to_add;
  std::unordered_set<Node*> eval_nodes_to_add;

  Node* shape_and_slice_node;
  TF_RETURN_IF_ERROR(save_node_vec[0]->input_node(2, &shape_and_slice_node));
  Tensor shape_and_slice_t;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(shape_and_slice_node->attrs(), "value", &shape_and_slice_t));
  Node* tensor_name_node;
  TF_RETURN_IF_ERROR(save_node_vec[0]->input_node(1, &tensor_name_node));
  Tensor tensor_name_t;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(tensor_name_node->attrs(), "value", &tensor_name_t));
  for (int k = 0; k < tensor_name_t.dim_size(0); ++k) {
    string tensor_n = tensor_name_t.flat<tstring>()(k);
    auto it = primary_node_metas_map.find(tensor_n);
    if (it == primary_node_metas_map.end()) continue;
    auto is_ev = it->second.m_var_type == VarType::EMBEDDING_VAR;
    if (!is_ev) {
      auto s_and_s_s = shape_and_slice_t.flat<tstring>()(k);
      std::vector<string> splits = str_util::Split(s_and_s_s, ' ');
      if (splits.size() < 2) {
        LOG(ERROR)
            << "Need least two elements in shape_and_slice specification: ";
      }
      std::vector<string> items =
          str_util::Split(splits.back(), ':', str_util::SkipEmpty());
      std::vector<int64> shape_vec(items.size() * 2, 1);
      for (int j = 0; j < items.size(); ++j) {
        int64 dim;
        if (!strings::safe_strto64(splits[j], &dim)) {
          LOG(ERROR) << "Non numerical dimension in shape_and_slice: ";
        }
        // partition_idx
        if (j == 0) {
          shape_vec[j] = dim;
          shape_vec[j + items.size()] = dim / cur_partition_nums_;
        } else {
          shape_vec[j] = dim;
          shape_vec[j + items.size()] = dim;
        }
      }
      variable_shape.emplace(tensor_n, std::move(shape_vec));
    }
  }

  auto rewrite_origin_shape_input = [this, &g, &save_node_vec, &nodes_to_delete,
                                     &node_to_origin_map, &primary_node_metas_map,
                                     &variable_shape, &nodes_to_add,
                                     &eval_nodes_to_add](int i) -> Status {
    Status s;
    Node* ori_save_node = save_node_vec[i];
    string assigned_device_name = ori_save_node->assigned_device_name();

    Node* tensor_names;
    TF_RETURN_IF_ERROR(ori_save_node->input_node(1, &tensor_names));
    nodes_to_delete.insert(tensor_names);
    Node* shape_and_slices;
    TF_RETURN_IF_ERROR(ori_save_node->input_node(2, &shape_and_slices));
    nodes_to_delete.insert(shape_and_slices);
    Tensor tensor_name_t;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(tensor_names->attrs(), "value", &tensor_name_t));
    Tensor shape_and_slice_t;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(shape_and_slices->attrs(), "value", &shape_and_slice_t));
    std::vector<string> new_tensor_shape;
    std::vector<string> new_tensor_name;
    for (int k = 0; k < tensor_name_t.dim_size(0); ++k) {
      string tensor_n = tensor_name_t.flat<tstring>()(k);
      auto s_and_s_s = shape_and_slice_t.flat<tstring>()(k);
      new_tensor_name.emplace_back(tensor_n);
      auto it = variable_shape.find(tensor_n);
      if (it == variable_shape.end()) {
        new_tensor_shape.emplace_back(s_and_s_s);
        LOG(INFO) << "tensor_name: " << tensor_n << " shape is: " << s_and_s_s;
        Node* input_node = nullptr;
        TF_RETURN_IF_ERROR(ori_save_node->input_node(5+k, &input_node));
        eval_nodes_to_add.emplace(input_node);
      } else {
        if ((primary_node_metas_map[tensor_n].m_var_type == VarType::DENSE_RESOUCE_VAR) ||
            (primary_node_metas_map[tensor_n].m_var_type == VarType::DENSE_REF_VAR) &&
            (primary_node_metas_map[tensor_n].m_partition_num != 1)) {
          {
            const Edge* e = nullptr;
            ori_save_node->input_edge(5+k, &e);// do not handle error
            if (e != nullptr) {
              g->RemoveEdge(e);
            }
          }
          
          for (auto* tmp_out: node_to_origin_map[tensor_n][i]->out_nodes()) {
            if (tmp_out->type_string() == "Identity") {
              g->AddEdge(tmp_out, 0, ori_save_node, 5+k); 
            }
          }
        }
        string tmp_shape_and_slice = "";
        auto shape_and_slice = it->second;
        for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
          tmp_shape_and_slice += std::to_string(shape_and_slice[j]);
          tmp_shape_and_slice += " ";
        }
        std::vector<string> tmp_dim;
        for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
          // partition_idx
          if (j == 0) {
            if ((!scaling_up_) && (i == cur_partition_nums_-1)) {
              int64 low = i * shape_and_slice[j + shape_and_slice.size() / 2];
              int64 high = shape_and_slice[j] - low;
              tmp_dim.emplace_back(std::to_string(low) + "," +
                                  std::to_string(high));
            } else {
              int64 low = i * shape_and_slice[j + shape_and_slice.size() / 2];
              int64 high = shape_and_slice[j + shape_and_slice.size() / 2];
              tmp_dim.emplace_back(std::to_string(low) + "," +
                                  std::to_string(high));
            }
          } else {
            int64 low = 0;
            int64 high = shape_and_slice[j];
            tmp_dim.emplace_back(std::to_string(low) + "," +
                                 std::to_string(high));
          }
        }
        tmp_shape_and_slice += str_util::Join(tmp_dim, ":");
        new_tensor_shape.emplace_back(tmp_shape_and_slice);
        LOG(INFO) << "tensor_name: " << tensor_n << " shape is: " << tmp_shape_and_slice;
      }
    }
    int old_tensor_size = new_tensor_shape.size();
    
    if ((i == 0) && (nodes_to_add.size() > 0)) {
      std::vector<DataType> n_dtypes;
      TF_RETURN_IF_ERROR(GetNodeAttr(ori_save_node->attrs(), "dtypes", &n_dtypes));
      int k = 0;
      for (auto& it:nodes_to_add) {
        new_tensor_name.emplace_back(it.second.first);
        new_tensor_shape.emplace_back(it.second.second);
        n_dtypes.emplace_back(DT_FLOAT);
        g->AddEdge(it.first, 0, ori_save_node, 5+old_tensor_size+k);
        ++k;
      }
      ori_save_node->ClearAttr("dtypes");
      ori_save_node->AddAttr("dtypes", n_dtypes);
    }

    int tensor_size = new_tensor_shape.size();
    // tensor_names
    Tensor new_tensor_name_t;
    TensorProto tensor_name_proto;
    tensor_name_proto.set_dtype(DT_STRING);
    TensorShape({tensor_size})
        .AsProto(tensor_name_proto.mutable_tensor_shape());
    for (int j = 0; j < new_tensor_name.size(); ++j) {
      tensor_name_proto.add_string_val(new_tensor_name[j]);
    }
    bool ret = new_tensor_name_t.FromProto(tensor_name_proto);
    if (!ret) return errors::Internal("tensor_name tensor init error");
    NodeDef name_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(tensor_names->name() + "/Copy", "Const")
            .Attr("value", new_tensor_name_t)
            .Attr("dtype", DT_STRING)
            .Finalize(&name_node_def));
    Node* name_node = g->AddNode(name_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    name_node->set_assigned_device_name(assigned_device_name);
    TF_RETURN_IF_ERROR(g->UpdateEdge(name_node, 0, ori_save_node, 1));

    Tensor new_tensor_shape_t;
    TensorProto tensor_shape_proto;
    tensor_shape_proto.set_dtype(DT_STRING);
    TensorShape({tensor_size})
        .AsProto(tensor_shape_proto.mutable_tensor_shape());
    for (int j = 0; j < new_tensor_shape.size(); ++j) {
      tensor_shape_proto.add_string_val(new_tensor_shape[j]);
    }
    ret = new_tensor_shape_t.FromProto(tensor_shape_proto);
    if (!ret) return errors::Internal("shape tensor init error");
    NodeDef shape_slice_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(shape_and_slices->name() + "/Copy", "Const")
            .Attr("value", new_tensor_shape_t)
            .Attr("dtype", DT_STRING)
            .Finalize(&shape_slice_node_def));
    Node* shape_slice_node = g->AddNode(shape_slice_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    shape_slice_node->set_assigned_device_name(assigned_device_name);
    TF_RETURN_IF_ERROR(g->UpdateEdge(shape_slice_node, 0, ori_save_node, 2));

    return s;
  };

  if (scaling_up_) {
    for (int i = 0; i < cur_partition_nums_ ; ++i) {
      if (i < save_node_vec.size()) {
        TF_RETURN_IF_ERROR(rewrite_origin_shape_input(i));
      } else {
        Status s;
        bool has_ev = false;
        Node* ori_save_node = save_node_vec[0];

        std::vector<Node*> kv_lookup_resource_node_vec;
        std::vector<DataType> n_dtypes;
        std::vector<string> tensor_names_vec;
        std::vector<DataType> ev_dtypes;
        std::vector<string> ev_names_vec;
        std::vector<DataType> key_data_types;
        std::vector<NodeDefBuilder::NodeOut> tensors_input;
        std::vector<Node*> tensor_vec;
        std::vector<Node*> restore_tensor_vec;
        std::string assigned_device_name = "";

        for (auto& it : primary_node_metas_map) {
          if (it.second.m_partition_num == 1) continue;
          auto ev_node = node_to_origin_map[it.first][i];
          bool is_ev = it.second.m_var_type == VarType::EMBEDDING_VAR;

          if (assigned_device_name == "")
            assigned_device_name = ev_node->assigned_device_name();
          if (is_ev) {
            has_ev = true;
            DataType key_type, value_type;
            TF_RETURN_IF_ERROR(
                GetNodeAttr(ev_node->attrs(), "Tkeys", &key_type));
            TF_RETURN_IF_ERROR(
                GetNodeAttr(ev_node->attrs(), "dtype", &value_type));
            NodeDef kv_lookup_resource_node_def;
            TF_RETURN_IF_ERROR(
                NodeDefBuilder(ev_node->name() + "/KvResourceLookupResource",
                               "KvResourceLookupResource")
                    .Input(ev_node->name(), 0, ev_node->output_type(0))
                    .Attr("Tkeys", key_type)
                    .Attr("dtype", value_type)
                    .Device(ev_node->assigned_device_name())
                    .Finalize(&kv_lookup_resource_node_def));
            Node* kv_lookup_resource_node =
                g->AddNode(kv_lookup_resource_node_def, &s);
            TF_RETURN_IF_ERROR(s);

            kv_lookup_resource_node_vec.emplace_back(kv_lookup_resource_node);
            ev_names_vec.emplace_back(ev_node->name());
            key_data_types.emplace_back(key_type);
          } else {
            tensor_names_vec.emplace_back(it.first);
            restore_tensor_vec.emplace_back(ev_node);
            for (auto* o_node : ev_node->out_nodes()) {
              if (o_node->type_string() == "Identity") {
                tensor_vec.emplace_back(o_node);
              }
            }
          }
        }

        if (has_ev) {
          // global_step
          Node* global_step_node;
          TF_RETURN_IF_ERROR(ori_save_node->input_node(5, &global_step_node));
          tensors_input.emplace_back(NodeDefBuilder::NodeOut(
              global_step_node->name(), 0, global_step_node->output_type(0)));
          n_dtypes.emplace_back(DT_INT64);
        }

        for (auto& tensor : tensor_vec) {
          tensors_input.emplace_back(NodeDefBuilder::NodeOut(
              tensor->name(), 0, tensor->output_type(0)));
          DataType t_type;
          TF_RETURN_IF_ERROR(GetNodeAttr(tensor->attrs(), "T", &t_type));
          n_dtypes.emplace_back(t_type);
        }

        Node* new_sharded_filename;
        Node* tensor_name_node;
        Node* shape_slice_node;
        Node* ev_name_node;
        Node* kv_lookup_resource_node;

        {
          Node* sharded_filename;
          TF_RETURN_IF_ERROR(ori_save_node->input_node(0, &sharded_filename));
          new_sharded_filename =
              CopyNode(g, sharded_filename, assigned_device_name, i);

          Node* prefix_name;
          TF_RETURN_IF_ERROR(sharded_filename->input_node(0, &prefix_name));
          g->AddEdge(prefix_name, 0, new_sharded_filename, 0);

          Node* num_shards;
          TF_RETURN_IF_ERROR(sharded_filename->input_node(2, &num_shards));
          Tensor new_tensor_nums(DT_INT32, TensorShape({}));
          new_tensor_nums.flat<int32>()(0) = cur_partition_nums_;
          num_shards->ClearAttr("value");
          num_shards->AddAttr("value", new_tensor_nums);
          g->AddEdge(num_shards, 0, new_sharded_filename, 2);

          Node* id_shards;
          TF_RETURN_IF_ERROR(sharded_filename->input_node(1, &id_shards));
          Node* new_id_shards = CopyNode(g, id_shards, assigned_device_name, i);
          Tensor new_tensor_ids(DT_INT32, TensorShape({}));
          new_tensor_ids.flat<int32>()(0) = i;
          new_id_shards->ClearAttr("value");
          new_id_shards->AddAttr("value", new_tensor_ids);
          g->AddEdge(new_id_shards, 0, new_sharded_filename, 1);
        }

        {
          int tensor_size =
              has_ev ? tensor_names_vec.size() + 1 : tensor_names_vec.size();
          // tensor_names
          Tensor new_tensor_names, new_tensor_shape;
          TensorProto tensor_shape_proto, tensor_name_proto;
          tensor_name_proto.set_dtype(DT_STRING);
          tensor_shape_proto.set_dtype(DT_STRING);
          TensorShape({tensor_size})
              .AsProto(tensor_shape_proto.mutable_tensor_shape());
          TensorShape({tensor_size})
              .AsProto(tensor_name_proto.mutable_tensor_shape());
          if (has_ev) {
            tensor_name_proto.add_string_val("global_step");
            tensor_shape_proto.add_string_val("");
          }
          for (int j = 0; j < tensor_names_vec.size(); ++j) {
            tensor_name_proto.add_string_val(tensor_names_vec[j]);
          }
          new_tensor_names.FromProto(tensor_name_proto);
          NodeDef tensor_name_node_def;
          TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                                std::to_string(i) +
                                                "/tensor_names",
                                            "Const")
                                 .Attr("value", new_tensor_names)
                                 .Attr("dtype", DT_STRING)
                                 .Device(assigned_device_name)
                                 .Finalize(&tensor_name_node_def));
          tensor_name_node = g->AddNode(tensor_name_node_def, &s);
          TF_RETURN_IF_ERROR(s);

          std::vector<string> new_tensor_shape_vec;
          for (int j = 0; j < tensor_names_vec.size(); ++j) {
            string tensor_n = tensor_names_vec[j];
            LOG(INFO) << "cur tensor_n is : " << tensor_n;
            auto it = variable_shape.find(tensor_n);
            if (it != variable_shape.end()) {
              string tmp_shape_and_slice = "";
              auto shape_and_slice = it->second;
              for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
                tmp_shape_and_slice += std::to_string(shape_and_slice[j]);
                tmp_shape_and_slice += " ";
              }
              std::vector<string> tmp_dim;
              for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
                // partition_idx
                if (j == 0) {
                  int64 low =
                      i * shape_and_slice[j + shape_and_slice.size() / 2];
                  int64 high;
                  if (i == cur_partition_nums_ - 1) {
                    high = shape_and_slice[j] - low;
                  } else {
                    high = shape_and_slice[j + shape_and_slice.size() / 2];
                  }
                  tmp_dim.emplace_back(std::to_string(low) + "," +
                                       std::to_string(high));
                } else {
                  int64 low = 0;
                  int64 high = shape_and_slice[j];
                  tmp_dim.emplace_back(std::to_string(low) + "," +
                                       std::to_string(high));
                }
              }
              tmp_shape_and_slice += str_util::Join(tmp_dim, ":");
              LOG(INFO) << "tmp_shape_and_slice is: " << tmp_shape_and_slice;
              tensor_shape_proto.add_string_val(tmp_shape_and_slice);
            }
          }
          new_tensor_shape.FromProto(tensor_shape_proto);
          LOG(INFO) << " processing shape_slice_node...";
          NodeDef shape_slice_node_def;
          TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                                std::to_string(i) +
                                                "/shape_and_slices",
                                            "Const")
                                 .Attr("value", new_tensor_shape)
                                 .Attr("dtype", DT_STRING)
                                 .Device(assigned_device_name)
                                 .Finalize(&shape_slice_node_def));
          shape_slice_node = g->AddNode(shape_slice_node_def, &s);
          TF_RETURN_IF_ERROR(s);
        }

        {
          LOG(INFO) << " processing ev_name_node...";
          // ev_names
          NodeDef ev_name_node_def;
          Tensor ev_names_tensor;
          TensorProto ev_names_proto;
          ev_names_proto.set_dtype(DT_STRING);
          TensorShape({static_cast<int64>(ev_names_vec.size())})
              .AsProto(ev_names_proto.mutable_tensor_shape());
          for (int k = 0; k < ev_names_vec.size(); ++k) {
            ev_names_proto.add_string_val(ev_names_vec[k]);
          }
          ev_names_tensor.FromProto(ev_names_proto);
          TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                                std::to_string(i) + "/ev_names",
                                            "Const")
                                 .Attr("value", ev_names_tensor)
                                 .Attr("dtype", DT_STRING)
                                 .Device(assigned_device_name)
                                 .Finalize(&ev_name_node_def));
          ev_name_node = g->AddNode(ev_name_node_def, &s);
          TF_RETURN_IF_ERROR(s);
        }

        {
          LOG(INFO) << " processing kv_lookup_resource_node...";
          std::vector<NodeDefBuilder::NodeOut> kv_lookup_resource_input;
          for (auto* n : kv_lookup_resource_node_vec) {
            kv_lookup_resource_input.emplace_back(n->name(), 0,
                                                  n->output_type(0));
            ev_dtypes.emplace_back(DT_INT64);
          }
          DataType key_type;
          if (key_data_types.size() == 0) {
            key_type = DT_INT64;
            Tensor const_tensor(DT_INT64, TensorShape({}));
            // ev_resources
            NodeDef const_node_def;
            TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                                  std::to_string(i) +
                                                  "/ev_resources",
                                              "Const")
                                   .Attr("dtype", key_type)
                                   .Attr("value", const_tensor)
                                   .Device(assigned_device_name)
                                   .Finalize(&const_node_def));
            kv_lookup_resource_node = g->AddNode(const_node_def, &s);
            TF_RETURN_IF_ERROR(s);
          } else {
            key_type = key_data_types[0];
            // ev_resources
            NodeDef kv_lookup_resource_node_def;
            int n = kv_lookup_resource_node_vec.size();
            TF_RETURN_IF_ERROR(NodeDefBuilder(ori_save_node->name() + "_" +
                                                  std::to_string(i) +
                                                  "/ev_resources",
                                              "Pack")
                                   .Input(kv_lookup_resource_input)
                                   .Attr("N", n)
                                   .Attr("T", key_type)
                                   .Attr("axis", 0)
                                   .Device(assigned_device_name)
                                   .Finalize(&kv_lookup_resource_node_def));
            kv_lookup_resource_node =
                g->AddNode(kv_lookup_resource_node_def, &s);
            TF_RETURN_IF_ERROR(s);
          }
        }

        // tensor_names
        NodeDef save_node_def;
        TF_RETURN_IF_ERROR(
            NodeDefBuilder(ori_save_node->name() + "_" + std::to_string(i),
                           kSaveOp)
                .Input(new_sharded_filename->name(), 0,
                       new_sharded_filename->output_type(0))
                .Input(tensor_name_node->name(), 0,
                       tensor_name_node->output_type(0))
                .Input(shape_slice_node->name(), 0,
                       shape_slice_node->output_type(0))
                .Input(ev_name_node->name(), 0, ev_name_node->output_type(0))
                .Input(kv_lookup_resource_node->name(), 0,
                       kv_lookup_resource_node->output_type(0))
                .Input(tensors_input)
                .Attr("dtypes", n_dtypes)
                .Attr("ev_key_types", ev_dtypes)
                .Attr("has_ev", has_ev)
                .Finalize(&save_node_def));
        Node* save_node = g->AddNode(save_node_def, &s);
        TF_RETURN_IF_ERROR(s);
        save_node->set_assigned_device_name(assigned_device_name);

        for (auto* o_edge : ori_save_node->out_edges()) {
          if (o_edge->IsControlEdge()) {
            Node* save_control_node =
                CopyNode(g, o_edge->dst(), assigned_device_name, i);
            g->AddEdge(new_sharded_filename, 0, save_control_node, 0);
            g->AddControlEdge(save_node, save_control_node);
            for (auto* oo_edge : o_edge->dst()->out_edges()) {
              if (oo_edge->IsControlEdge()) {
                auto* dst_node = oo_edge->dst();
                g->AddControlEdge(save_control_node, dst_node);
                if (dst_node->type_string() == "Pack") {
                  int part_num;
                  TF_RETURN_IF_ERROR(
                      GetNodeAttr(dst_node->attrs(), "N", &part_num));
                  if (part_num != cur_partition_nums_) {
                    dst_node->ClearAttr("N");
                    dst_node->AddAttr("N", cur_partition_nums_);
                  }
                  g->AddEdge(new_sharded_filename, 0, dst_node, i);
                }
              }
            }
          }
        }
        Node* ori_restore_node = restore_node_vec[0];
        Node* restore_tensor_name_node;
        Node* restore_shape_slice_node;
        {
          int tensor_size = tensor_names_vec.size();
          // tensor_names
          Tensor new_tensor_names, new_tensor_shape;
          TensorProto tensor_shape_proto, tensor_name_proto;
          tensor_name_proto.set_dtype(DT_STRING);
          tensor_shape_proto.set_dtype(DT_STRING);
          TensorShape({tensor_size})
              .AsProto(tensor_shape_proto.mutable_tensor_shape());
          TensorShape({tensor_size})
              .AsProto(tensor_name_proto.mutable_tensor_shape());
          for (int j = 0; j < tensor_names_vec.size(); ++j) {
            tensor_name_proto.add_string_val(tensor_names_vec[j]);
          }
          new_tensor_names.FromProto(tensor_name_proto);
          NodeDef tensor_name_node_def;
          TF_RETURN_IF_ERROR(NodeDefBuilder(ori_restore_node->name() + "_" +
                                                std::to_string(i) +
                                                "/tensor_names",
                                            "Const")
                                  .Attr("value", new_tensor_names)
                                  .Attr("dtype", DT_STRING)
                                  .Finalize(&tensor_name_node_def));
          restore_tensor_name_node = g->AddNode(tensor_name_node_def, &s);
          TF_RETURN_IF_ERROR(s);
          restore_tensor_name_node->set_assigned_device_name(assigned_device_name);
          std::vector<string> new_tensor_shape_vec;
          for (int j = 0; j < tensor_names_vec.size(); ++j) {
            string tensor_n = tensor_names_vec[j];
            auto it = variable_shape.find(tensor_n);
            if (it != variable_shape.end()) {
              string tmp_shape_and_slice = "";
              auto shape_and_slice = it->second;
              for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
                tmp_shape_and_slice += std::to_string(shape_and_slice[j]);
                tmp_shape_and_slice += " ";
              }
              std::vector<string> tmp_dim;
              for (int j = 0; j < shape_and_slice.size() / 2; ++j) {
                // partition_idx
                if (j == 0) {
                  int64 low =
                      i * shape_and_slice[j + shape_and_slice.size() / 2];
                  int64 high;
                  if (i == cur_partition_nums_ - 1) {
                    high = shape_and_slice[j] - low;
                  } else {
                    high = shape_and_slice[j + shape_and_slice.size() / 2];
                  }
                  
                  tmp_dim.emplace_back(std::to_string(low) + "," +
                                        std::to_string(high));
                } else {
                  int64 low = 0;
                  int64 high = shape_and_slice[j];
                  tmp_dim.emplace_back(std::to_string(low) + "," +
                                        std::to_string(high));
                }
              }
              tmp_shape_and_slice += str_util::Join(tmp_dim, ":");
              LOG(INFO) << "tmp_shape_and_slice is: " << tmp_shape_and_slice;
              tensor_shape_proto.add_string_val(tmp_shape_and_slice);
            }
          }
          new_tensor_shape.FromProto(tensor_shape_proto);
          LOG(INFO) << " processing shape_slice_node...";
          NodeDef shape_slice_node_def;
          TF_RETURN_IF_ERROR(NodeDefBuilder(ori_restore_node->name() + "_" +
                                                std::to_string(i) +
                                                "/shape_and_slices",
                                            "Const")
                                  .Attr("value", new_tensor_shape)
                                  .Attr("dtype", DT_STRING)
                                  .Finalize(&shape_slice_node_def));
          restore_shape_slice_node = g->AddNode(shape_slice_node_def, &s);
          TF_RETURN_IF_ERROR(s);
          restore_shape_slice_node->set_assigned_device_name(assigned_device_name);
        }

        NodeDef restore_node_def;
        if (has_ev) {
          n_dtypes.erase(n_dtypes.begin());
        }
        TF_RETURN_IF_ERROR(
            NodeDefBuilder(ori_restore_node->name() + "_" + std::to_string(i),
                            kRestoreOp)
                .Input(new_sharded_filename->name(), 0,
                        new_sharded_filename->output_type(0))
                .Input(restore_tensor_name_node->name(), 0,
                        restore_tensor_name_node->output_type(0))
                .Input(restore_shape_slice_node->name(), 0,
                        restore_shape_slice_node->output_type(0))
                .Attr("dtypes", n_dtypes)
                .Finalize(&restore_node_def));
        Node* restore_node = g->AddNode(restore_node_def, &s);
        TF_RETURN_IF_ERROR(s); 
        restore_node->set_assigned_device_name(assigned_device_name);

        NodeDef restore_no_op_def;
        TF_RETURN_IF_ERROR(
            NodeDefBuilder("save/restore_all/NoOp_" + std::to_string(i),
                            "NoOp")
                .Finalize(&restore_no_op_def));
        Node* restore_no_op = g->AddNode(restore_no_op_def, &s);
        TF_RETURN_IF_ERROR(s);
        restore_no_op->set_assigned_device_name(assigned_device_name);

        for (int k = 0; k < restore_tensor_vec.size(); ++k) {
          LOG(INFO) << restore_tensor_vec[k]->name() << " === "
                    << tensor_names_vec[k];
          for (auto* o_node: restore_tensor_vec[k]->out_nodes()) {
            if (o_node->type_string() == "Assign") {
      	      Node* restore_assign_node =
                CopyNode(g, o_node, assigned_device_name, i, o_node->name()+"/Copy");
	            g->AddEdge(restore_tensor_vec[k], 0, restore_assign_node, 0);
              g->AddEdge(restore_node, k, restore_assign_node, 1);
              g->AddControlEdge(restore_assign_node, restore_no_op);
            }
          }
        }
        for (auto* n: g->nodes()) {
          if (n->name() == "save/restore_all") {
            g->AddControlEdge(restore_no_op, n);
            break;
          }
        }
      }
    }
  } else if (save_node_vec.size() > cur_partition_nums_) {
    for (int i = save_node_vec.size() - 1; i >=0 ; --i) {
      Node* cur_save_node = save_node_vec[i];
      if (i < cur_partition_nums_) {
        TF_RETURN_IF_ERROR(rewrite_origin_shape_input(i));
      } else {
        {
          Node* sharded_filename;
          TF_RETURN_IF_ERROR(cur_save_node->input_node(0, &sharded_filename));
          for (auto* o_node : sharded_filename->out_nodes()) {
            if (o_node->type_string() == kIdentityOp) {
              nodes_to_delete.insert(o_node);
            } else if (o_node->type_string() == "Pack") {
              int part_num;
              TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "N", &part_num));
              if (part_num != cur_partition_nums_) {
                o_node->ClearAttr("N");
                o_node->AddAttr("N", cur_partition_nums_);
              }
            }
          }
          nodes_to_delete.insert(sharded_filename);
        }

        {
          Node* tensor_names;
          TF_RETURN_IF_ERROR(cur_save_node->input_node(1, &tensor_names));
          Tensor tensor_name_t;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(tensor_names->attrs(), "value", &tensor_name_t));
          Node* shape_and_slices;
          TF_RETURN_IF_ERROR(cur_save_node->input_node(2, &shape_and_slices));
          Tensor shape_and_slice_t;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(shape_and_slices->attrs(), "value", &shape_and_slice_t));
          for (int k = 0; k < tensor_name_t.dim_size(0); ++k) {
            string tensor_n = tensor_name_t.flat<tstring>()(k);
            string s_and_s_s = shape_and_slice_t.flat<tstring>()(k);
            auto it = variable_shape.find(tensor_n);
            if (it == variable_shape.end()) {
              if (tensor_n != "global_step") {
		            for (auto* n: cur_save_node->in_nodes()) {
                  if (n->name() == tensor_n) {
            	      nodes_to_add.emplace(n, std::pair<string, string>(tensor_n, s_and_s_s));
          	      }
		            }
              }
            }
          }
          nodes_to_delete.insert(tensor_names);
          nodes_to_delete.insert(shape_and_slices);
        }

        {
          Node* ev_names;
          TF_RETURN_IF_ERROR(cur_save_node->input_node(3, &ev_names));
          nodes_to_delete.insert(ev_names);
          Node* ev_resource;
          TF_RETURN_IF_ERROR(cur_save_node->input_node(4, &ev_resource));
          nodes_to_delete.insert(ev_resource);
        }
        nodes_to_delete.insert(cur_save_node);
      }
    }
    TF_RETURN_IF_ERROR(MoveUnPartitionedVariable(g, unpartitioned_node_map, 
                                                 nodes_to_add, eval_nodes_to_add, meta_node));
  }

  for (auto* n : nodes_to_delete) {
    g->RemoveNode(n);
  }

  return Status::OK();
}

Status ElasticTrainingPass::RewriteTrainingSubGraph(
    Graph* g, std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
    std::unordered_map<std::string, std::vector<std::string>>&
        primary_node_to_opt_map,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    ElasticHookMetaNode& meta_node, bool is_test) {
  std::vector<Node*> no_op_vec(cur_partition_nums_, nullptr);

  for (auto it : primary_node_to_opt_map) {
    std::unordered_set<Node*> nodes_to_delete;

    auto& primary_variable_name = it.first;
    auto& opt_ev_names = it.second;
    int ev_partition_num = primary_node_metas_map[primary_variable_name].m_partition_num;
    VarType var_type = primary_node_metas_map[primary_variable_name].m_var_type;
    int part_var_full_shape = primary_node_metas_map[primary_variable_name].m_full_shape;
    auto& var_vec = node_to_origin_map[primary_variable_name];

    // var_vec.erase(std::remove(var_vec.begin(), var_vec.end(), nullptr), var_vec.end());

    // Make sure the opt variable is sorted by part.
    std::sort(opt_ev_names.begin(), opt_ev_names.end(),
              [](const std::string& str1, const std::string& str2) {
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

                return std::stoi(post_str.substr(post_idx)) <
                       std::stoi(post_str_1.substr(post_idx_1));
              }); 

    LOG(INFO) << "processing: " << primary_variable_name 
              <<  " var_type " << var_type
              << " var_vec size: " << var_vec.size();

    Node* elastic_node;
    Node* p_dynamic_stitch_node;

    // TODO(JUNQI) : per variable placement strategy
    if ((ev_partition_num == cur_partition_nums_) || (ev_partition_num == 1)) {
      LOG(INFO) << "Skip current variable.";
      continue;
    } else if (ev_partition_num < cur_partition_nums_) {

      TF_RETURN_IF_ERROR(ScalingUpForWardGraph(var_type, g, node_to_origin_map, 
                                               nodes_to_delete, primary_variable_name,
                                               opt_ev_names, meta_node, ev_partition_num));

      std::vector<Node*> primary_ev_filters(cur_partition_nums_, nullptr);
      TF_RETURN_IF_ERROR(ScalingUpRedistributionGraph(
          var_type, g, var_vec, meta_node.m_import_op_main, ev_partition_num, primary_ev_filters));
      for (auto& opt_ev_name : opt_ev_names) {
        TF_RETURN_IF_ERROR(ScalingUpRedistributionGraph(
            var_type, g, node_to_origin_map[opt_ev_name], meta_node.m_import_op_main,
            ev_partition_num, primary_ev_filters));
      }
      TF_RETURN_IF_ERROR(RewriteElasticPartitionGraph(
          var_type, g, node_to_origin_map[primary_variable_name], &elastic_node,
          &p_dynamic_stitch_node, nodes_to_delete));
      TF_RETURN_IF_ERROR(ScalingUpBackWardGraph(
          var_type, g, node_to_origin_map, primary_variable_name, opt_ev_names,
          elastic_node, p_dynamic_stitch_node, no_op_vec, part_var_full_shape,
          ev_partition_num));
          
    } else {  // scale down

      TF_RETURN_IF_ERROR(ScalingDownForWardGraph(var_type, g, node_to_origin_map, nodes_to_delete,
                                                 primary_variable_name, opt_ev_names, ev_partition_num));

      TF_RETURN_IF_ERROR(
              ScalingDownRedistributionGraph(var_type, g, var_vec, nodes_to_delete, ev_partition_num));
      for (auto& opt_ev_name : opt_ev_names) {
        TF_RETURN_IF_ERROR(ScalingDownRedistributionGraph(
            var_type, g, node_to_origin_map[opt_ev_name], nodes_to_delete, ev_partition_num));
      }

      TF_RETURN_IF_ERROR(RewriteElasticPartitionGraph(
              var_type, g, node_to_origin_map[primary_variable_name], &elastic_node,
              &p_dynamic_stitch_node, nodes_to_delete));

      TF_RETURN_IF_ERROR(ScalingDownBackWardGraph(
          g, var_type, node_to_origin_map, nodes_to_delete,
          primary_variable_name, opt_ev_names, 
          elastic_node, p_dynamic_stitch_node, part_var_full_shape, ev_partition_num));
    }

    for (auto* node : nodes_to_delete) {
      g->RemoveNode(node);
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpForWardGraph(const VarType& var_type, Graph* g, 
                                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                                  std::unordered_set<Node*>& nodes_to_delete,
                                                  const std::string& primary_variable_name,
                                                  const std::vector<std::string>& opt_ev_names,
                                                  ElasticHookMetaNode& meta_node, int ev_partition_num) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      s = ScalingUpEVForWardGraph(var_type, g, node_to_origin_map, nodes_to_delete,
                              primary_variable_name, opt_ev_names, meta_node, ev_partition_num);
      break;
    case VarType::RESOURCE_VAR:
    case VarType::DENSE_RESOUCE_VAR:
      s = ScalingUpResVarForWardGraph(var_type, g, node_to_origin_map, nodes_to_delete,
                                  primary_variable_name, opt_ev_names, meta_node, ev_partition_num);
      break;
    default:
      s = ScalingUpVarForWardGraph(var_type, g, node_to_origin_map, nodes_to_delete,
                              primary_variable_name, opt_ev_names, meta_node, ev_partition_num);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingUpResVarForWardGraph(const VarType& var_type,
                                                  Graph* g, 
                                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                                  std::unordered_set<Node*>& nodes_to_delete,
                                                  const std::string& primary_variable_name,
                                                  const std::vector<std::string>& opt_ev_names,
                                                  ElasticHookMetaNode& meta_node, int ev_partition_num) {
  auto& var_vec = node_to_origin_map[primary_variable_name];
  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    Node* var_node = var_vec[0];
    Node* cur_init_op = meta_node.m_init_op_vec[i];
    string new_device_name = NewDeviceName(var_node, i);
    std::string op_name =
        primary_variable_name + "/" + kPart + std::to_string(i);
    Node* new_var_node = CopyNode(g, var_node, new_device_name, i, op_name);
    var_vec[i] = new_var_node;
    LOG(INFO) << "JUNQI resource variable ===>" << primary_variable_name
              << " === " << i;
    bool is_init = false;
    TF_RETURN_IF_ERROR(FindNode(var_node, "AssignVariableOp",
        [this, &g, &is_init, &new_var_node, &cur_init_op,
            new_device_name, i](Node* target_node) {
          if (!is_init) {
            is_init = true;
            Node* new_var_init = CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_var_node, 0, new_var_init, 0);

            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(target_node->input_edge(1, &init_value_edge));
            Node* new_init_value =
                CopyNode(g, init_value_edge->src(), new_device_name, i);
            g->AddEdge(new_init_value, 0, new_var_init, 1);
            g->AddControlEdge(new_var_init, cur_init_op);
          }
          return Status::OK();
        }));
    
    TF_RETURN_IF_ERROR(FindNode(var_node, "ResourceGather",
      [this, &g, &new_var_node, new_device_name, i](Node* target_node) {
        Node* gather_op = CopyNode(g, target_node, new_device_name, i);
        g->AddEdge(new_var_node, 0, gather_op, 0);
        const Edge* gather_id_edge = nullptr;
        TF_RETURN_IF_ERROR(target_node->input_edge(1, &gather_id_edge));
        g->AddEdge(gather_id_edge->src(), gather_id_edge->src_output(),
                    gather_op, 1);
        for (auto* o_edge : target_node->out_edges()) {
          if (o_edge->dst()->type_string() == kIdentityOp) {
            Node* identity_op = CopyNode(g, o_edge->dst(), new_device_name, i);
            g->AddEdge(gather_op, 0, identity_op, 0);
          }
        }
        return Status::OK();
      }));

    TF_RETURN_IF_ERROR(FindNode(var_node, "ReadVariableOp",
      [this, &g, &new_var_node, new_device_name, i](Node* target_node) {
        Node* new_var_read = CopyNode(g, target_node, new_device_name, i);
        g->AddEdge(new_var_node, 0, new_var_read, 0);
        for (auto* oo_node : target_node->out_nodes()) {
          // Normal Variable
          if (oo_node->type_string() == "ConcatV2") {
            int N;
            TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
            if (N != cur_partition_nums_) {
              const Edge* axis_edge;
              TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
              oo_node->ClearAttr("N");
              oo_node->AddAttr("N", cur_partition_nums_);
              g->RemoveEdge(axis_edge);
              g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_nums_);
            }
            g->AddEdge(new_var_read, 0, oo_node, i);
          }
        }
        return Status::OK();
      }));

    for (auto& opt_ev_name : opt_ev_names) {
      auto var_node = node_to_origin_map[opt_ev_name][0];
      auto sep_idx = opt_ev_name.rfind("/");
      std::string op_name = opt_ev_name.substr(0, sep_idx) + "/" + kPart +
                            std::to_string(i) +
                            opt_ev_name.substr(sep_idx);

      Node* new_opt_var_node =
          CopyNode(g, var_node, new_device_name, i, op_name);

      LOG(INFO) << "JUNQI  BACKWARD VAR ===>" << opt_ev_name
                << " === " << i;
      node_to_origin_map[opt_ev_name][i] = new_opt_var_node;
      is_init = false;

      TF_RETURN_IF_ERROR(FindNode(var_node, "AssignVariableOp",
      [this, &g, &is_init, &new_opt_var_node, &cur_init_op,
            new_device_name, i](Node* target_node) {
        if (!is_init) {
          is_init = true;
          Node* new_var_init = CopyNode(g, target_node, new_device_name, i);
          g->AddEdge(new_opt_var_node, 0, new_var_init, 0);

          const Edge* init_value_edge = nullptr;
          TF_RETURN_IF_ERROR(target_node->input_edge(1, &init_value_edge));
          Node* new_const_init =
              CopyNode(g, init_value_edge->src(), new_device_name, i);
          g->AddEdge(new_const_init, 0, new_var_init, 1);
          g->AddControlEdge(new_var_init, cur_init_op);
        }
        return Status::OK();
      }));

      TF_RETURN_IF_ERROR(FindNode(var_node, "ReadVariableOp",
      [this, &g, &new_opt_var_node, new_device_name, i](Node* target_node) {
        Node* new_var_read = CopyNode(g, target_node, new_device_name, i);
        g->AddEdge(new_opt_var_node, 0, new_var_read, 0);
        for (auto* oo_node : target_node->out_nodes()) {
          // Normal Variable
          if (oo_node->type_string() == "ConcatV2") {
            int N;
            TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
            if (N != cur_partition_nums_) {
              const Edge* axis_edge;
              TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
              oo_node->ClearAttr("N");
              oo_node->AddAttr("N", cur_partition_nums_);
              g->RemoveEdge(axis_edge);
              g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_nums_);
            }
            g->AddEdge(new_var_read, 0, oo_node, i);
          }
        }
        return Status::OK();
      }));

    }          
  }        
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpVarForWardGraph(const VarType& var_type,
                                                  Graph* g, 
                                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                                  std::unordered_set<Node*>& nodes_to_delete,
                                                  const std::string& primary_variable_name,
                                                  const std::vector<std::string>& opt_ev_names,
                                                  ElasticHookMetaNode& meta_node, int ev_partition_num) {
  auto& var_vec = node_to_origin_map[primary_variable_name];
  Node* var_node = var_vec[0];
  //Change read graph according to tensorflow/python/ops/variables.py
  if (ev_partition_num == 1) {
    TF_RETURN_IF_ERROR(FindNode(var_node, "Identity",
        [this, &var_node, &g](Node* target_node) {
      Status s;
      Tensor axis_tensor(DT_INT32, TensorShape({}));
      axis_tensor.flat<int>()(0) = 0;
      NodeDef axis_def;
      TF_RETURN_IF_ERROR(
          NodeDefBuilder(var_node->name() + "ConcatPartitions/concat/axis",
                        "Const")
              .Attr("dtype", DT_INT32)
              .Attr("value", axis_tensor)
              .Finalize(&axis_def));
      Node* axis_node = g->AddNode(axis_def, &s);
      TF_RETURN_IF_ERROR(s);
      axis_node->set_assigned_device_name(var_node->assigned_device_name());

      NodeDef concat_node_def;
      TF_RETURN_IF_ERROR(
          NodeDefBuilder(var_node->name() + "ConcatPartitions/concat",
                        "ConcatV2")
              .Input(target_node->name(), 0, target_node->output_type(0))
              .Input(axis_node->name(), 0, axis_node->output_type(0))
              .Attr("N", 1)
              .Attr("T", target_node->output_type(0))
              .Finalize(&concat_node_def));
      Node* concat_node = g->AddNode(concat_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      concat_node->set_assigned_device_name(var_node->assigned_device_name());
      for (auto* o_edge: target_node->out_edges()) {
        TF_RETURN_IF_ERROR(g->UpdateEdge(concat_node, 0, o_edge->dst(), o_edge->dst_input()));
      }
      return s;
      }));
  }

  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    Node* cur_init_op = meta_node.m_init_op_vec[i];
    string new_device_name = NewDeviceName(var_node, i);
    std::string op_name =
        primary_variable_name + "/" + kPart + std::to_string(i);
    Node* new_var_node = CopyNode(g, var_node, new_device_name, i, op_name);
    var_vec[i] = new_var_node;
    bool is_init = false;
    TF_RETURN_IF_ERROR(FindNode(var_node, "Assign",
        [this, &g, &is_init, &new_var_node, &cur_init_op,
            new_device_name, i](Node* target_node) {
          if (!is_init) {
            is_init = true;
            Node* new_var_init = CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_var_node, 0, new_var_init, 0);

            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(target_node->input_edge(1, &init_value_edge));
            Node* new_init_value = CopyNode(g, init_value_edge->src(), new_device_name, i);
            g->AddEdge(new_init_value, 0, new_var_init, 1);
            g->AddControlEdge(new_var_init, cur_init_op);
          }
          return Status::OK();
        }));
    
    TF_RETURN_IF_ERROR(FindNode(var_node, kIdentityOp,
      [this, &g, &new_var_node, &var_type, new_device_name, i](Node* target_node) {
        Node* new_var_read = CopyNode(g, target_node, new_device_name, i);
        g->AddEdge(new_var_node, 0, new_var_read, 0);
        for (auto* oo_node : target_node->out_nodes()) {
          // Normal Variable
          if (oo_node->type_string() == "ConcatV2") {
            if (oo_node->name().find(kElasticImportScope) != string::npos) {
              continue;
            }
            // exactly once
            int N;
            TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
            if (N != cur_partition_nums_) {
              const Edge* axis_edge;
              TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
              oo_node->ClearAttr("N");
              oo_node->AddAttr("N", cur_partition_nums_);
              g->RemoveEdge(axis_edge);
              g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_nums_);
            }
            g->AddEdge(new_var_read, 0, oo_node, i);
          } else if (oo_node->type_string() == "GatherV2") {
            Node* new_gather = CopyNode(g, oo_node, new_device_name, i);
            g->AddEdge(new_var_read, 0, new_gather, 0);
            Node* axis_node = nullptr;
            TF_RETURN_IF_ERROR(oo_node->input_node(2, &axis_node));
            Node* new_axis_node =
                CopyNode(g, axis_node, new_device_name, i);
            g->AddEdge(new_axis_node, 0, new_gather, 2);
          }
        }
        return Status::OK();
      }));

    for (auto& opt_ev_name : opt_ev_names) {
      auto var_node = node_to_origin_map[opt_ev_name][0];
      auto sep_idx = opt_ev_name.rfind("/");
      std::string op_name = opt_ev_name.substr(0, sep_idx) + "/" + kPart +
                            std::to_string(i) +
                            opt_ev_name.substr(sep_idx);

      Node* new_opt_var_node =
          CopyNode(g, var_node, new_device_name, i, op_name);

      // LOG(INFO) << "JUNQI  BACKWARD VAR ===>" << opt_ev_name
      //           << " === " << i;
      node_to_origin_map[opt_ev_name][i] = new_opt_var_node;
      is_init = false;

      TF_RETURN_IF_ERROR(FindNode(var_node, "Assign",
      [this, &g, &is_init, &new_opt_var_node, &cur_init_op,
            new_device_name, i](Node* target_node) {
        if (!is_init) {
          is_init = true;
          Node* new_var_init = CopyNode(g, target_node, new_device_name, i);
          g->AddEdge(new_opt_var_node, 0, new_var_init, 0);

          const Edge* init_value_edge = nullptr;
          TF_RETURN_IF_ERROR(target_node->input_edge(1, &init_value_edge));
          Node* new_const_init =
              CopyNode(g, init_value_edge->src(), new_device_name, i);
          g->AddEdge(new_const_init, 0, new_var_init, 1);
          g->AddControlEdge(new_var_init, cur_init_op);
        }
        return Status::OK();
      }));

      TF_RETURN_IF_ERROR(FindNode(var_node, kIdentityOp,
      [this, &g, &new_opt_var_node, new_device_name, i](Node* target_node) {
        Node* new_opt_var_read = CopyNode(g, target_node, new_device_name, i);
        g->AddEdge(new_opt_var_node, 0, new_opt_var_read, 0);
        for (auto* oo_node : target_node->out_nodes()) {
          // Normal Variable
          if (oo_node->type_string() == "ConcatV2") {
            if (oo_node->name().find(kElasticImportScope) != string::npos) {
              continue;
            }
            int N;
            TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "N", &N));
            if (N != cur_partition_nums_) {
              const Edge* axis_edge;
              TF_RETURN_IF_ERROR(oo_node->input_edge(N, &axis_edge));
              oo_node->ClearAttr("N");
              oo_node->AddAttr("N", cur_partition_nums_);
              g->AddEdge(axis_edge->src(), 0, oo_node, cur_partition_nums_);
            }
            TF_RETURN_IF_ERROR(
                g->UpdateEdge(new_opt_var_read, 0, oo_node, i));
          }
        }
        return Status::OK();
      }));
    }          
  }        
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpEVForWardGraph(const VarType& var_type,
                                                  Graph* g, 
                                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                                  std::unordered_set<Node*>& nodes_to_delete,
                                                  const std::string& primary_variable_name,
                                                  const std::vector<std::string>& opt_ev_names,
                                                  ElasticHookMetaNode& meta_node, int ev_partition_num) {
  LOG(INFO) << "CALLING ScalingUpEVForWardGraph";
  // EVHandler
  auto& var_vec = node_to_origin_map[primary_variable_name];
  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    Node* cur_init_op = meta_node.m_init_op_vec[i];
    Node* ori_ev_node = var_vec[0];
    string new_device_name = NewDeviceName(ori_ev_node, i);
    std::string op_name =
        primary_variable_name + "/" + kPart + std::to_string(i);

    Node* new_ev_node = CopyNode(g, ori_ev_node, new_device_name, i, op_name);
    new_ev_node->ClearAttr("shared_name");
    new_ev_node->AddAttr("shared_name", op_name);
    var_vec[i] = new_ev_node;

    bool is_init = false;
    Node* primary_init_node;
    // InitializeEVResource
    TF_RETURN_IF_ERROR(FindNode(ori_ev_node, kEvInitOp, 
        [this, &primary_init_node, &g, &new_ev_node, &is_init,
          new_device_name, i](Node* target_node) {
          if (!is_init) {
            const Node* tmp_check_ev_0;
            TF_RETURN_IF_ERROR(target_node->input_node(0, &tmp_check_ev_0));
            const Node* tmp_check_ev_1;
            TF_RETURN_IF_ERROR(target_node->input_node(1, &tmp_check_ev_1));
            if (tmp_check_ev_0->name() == tmp_check_ev_1->name()) {
              is_init = true;
              primary_init_node =
                  CopyNode(g, target_node, new_device_name, i);
              g->AddEdge(new_ev_node, 0, primary_init_node, 0);
              g->AddEdge(new_ev_node, 0, primary_init_node, 1);
              // init_value
              const Edge* init_value_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(2, &init_value_edge));
              auto* init_value_node = CopyNode(
                  g, init_value_edge->src(), new_device_name, i);
              g->AddEdge(init_value_node, init_value_edge->src_output(),
                          primary_init_node, 2);

              // empty_key
              const Edge* empty_key_edge = nullptr;
              TF_RETURN_IF_ERROR(target_node->input_edge(3, &empty_key_edge));
              auto* empty_key_node = CopyNode(
                  g, empty_key_edge->src(), new_device_name, i);
              g->AddEdge(empty_key_node, empty_key_edge->src_output(),
                          primary_init_node, 3);
            }
          }
          return Status::OK();
        }));

    g->AddControlEdge(primary_init_node, cur_init_op);

    TF_RETURN_IF_ERROR(FindNode(ori_ev_node, "KvResourceGather", 
      [this, &g, &new_ev_node, new_device_name, i](Node* target_node) {
        Node* gather_op = CopyNode(g, target_node, new_device_name, i);
        g->AddEdge(new_ev_node, 0, gather_op, 0);
        const Edge* gather_id_edge = nullptr;
        TF_RETURN_IF_ERROR(target_node->input_edge(1, &gather_id_edge));
        g->AddEdge(gather_id_edge->src(), gather_id_edge->src_output(),
                    gather_op, 1);
        const Edge* axis_edge = nullptr;
        TF_RETURN_IF_ERROR(target_node->input_edge(2, &axis_edge));
        Node* axis = CopyNode(g, axis_edge->src(), new_device_name, i);
        g->AddEdge(axis, 0, gather_op, 2);
        for (auto* o_edge : target_node->out_edges()) {
          if (o_edge->dst()->type_string() == kIdentityOp) {
            Node* identity_op = CopyNode(g, o_edge->dst(), new_device_name, i);
            g->AddEdge(gather_op, 0, identity_op, 0);
          }
        }
        return Status::OK();
      }));

    // OptEV
    for (auto& opt_ev_name : opt_ev_names) {
      auto opt_var_node = node_to_origin_map[opt_ev_name][0];
      auto sep_idx = opt_ev_name.rfind("/");
      std::string op_name = opt_ev_name.substr(0, sep_idx) + "/" + kPart +
                            std::to_string(i) +
                            opt_ev_name.substr(sep_idx);

      // EVHandler
      Node* new_opt_ev_node =
          CopyNode(g, opt_var_node, new_device_name, i, op_name);
      new_opt_ev_node->ClearAttr("shared_name");
      new_opt_ev_node->AddAttr("shared_name", op_name);
      node_to_origin_map[opt_ev_name][i] = new_opt_ev_node;

      is_init = false;
      TF_RETURN_IF_ERROR(FindNode(opt_var_node, kEvInitOp,
        [this, &g, &primary_init_node, &new_ev_node, &cur_init_op, &is_init,
          &new_opt_ev_node, new_device_name, i](Node* target_node) {
          if (!is_init) {
            is_init = true;
            Node* init_node =
                  CopyNode(g, target_node, new_device_name, i);
            g->AddEdge(new_opt_ev_node, 0, init_node, 0);
            g->AddEdge(new_ev_node, 0, init_node, 1);
            g->AddControlEdge(primary_init_node, init_node);
            g->AddControlEdge(init_node, cur_init_op);
            // init_value
            const Edge* init_value_edge = nullptr;
            TF_RETURN_IF_ERROR(target_node->input_edge(2, &init_value_edge));
            auto* init_value_node =
                CopyNode(g, init_value_edge->src(), new_device_name, i);
            g->AddEdge(init_value_node, init_value_edge->src_output(),
                        init_node, 2);

            // empty_key
            const Edge* empty_key_edge = nullptr;
            TF_RETURN_IF_ERROR(target_node->input_edge(3, &empty_key_edge));
            Node* empty_key_node =
                CopyNode(g, empty_key_edge->src(), new_device_name, i);

            g->AddEdge(empty_key_node, empty_key_edge->src_output(),
                        init_node, 3);
          }
          return Status::OK();
        }));
    }    
  }
  return Status::OK();                                                  
}

Status ElasticTrainingPass::ScalingDownResVarForWardGraph(const VarType& var_type, Graph* g, 
                                                          std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                                          std::unordered_set<Node*>& nodes_to_delete,
                                                          const std::string& primary_variable_name,
                                                          const std::vector<std::string>& opt_ev_names,
                                                          int ev_partition_num) {
  for (int i = 0; i < ev_partition_num; ++i) {
    Node* ev_node = node_to_origin_map[primary_variable_name][i];
    // if ( i < cur_partition_nums_) {
    //   LOG(INFO) << "JUNQI SCALEDOWN VAR===>" << primary_variable_name
    //             << " === " << i << " === " << ev_node->name();
    //   for (auto* o_node : ev_node->out_nodes()) {
    //     if (o_node->type_string() == kIdentityOp) {
    //       for (auto* o_edge : o_node->out_edges()) {
    //         // Normal Variable
    //         if (o_edge->dst()->type_string() == "ConcatV2") {
    //           int N;
    //           TF_RETURN_IF_ERROR(GetNodeAttr(o_edge->dst()->attrs(), "N", &N));
    //           if (N != cur_partition_nums_) {
    //             const Edge* axis_edge = nullptr;
    //             TF_RETURN_IF_ERROR(o_edge->dst()->input_edge(N, &axis_edge));
    //             g->AddEdge(axis_edge->src(), 0, o_edge->dst(), cur_partition_nums_);
    //             g->RemoveEdge(axis_edge);
    //             o_edge->dst()->ClearAttr("N");
    //             o_edge->dst()->AddAttr("N", cur_partition_nums_);
    //           }
    //           if (o_edge->dst_input() != i){
    //             g->AddEdge(o_node, 0, o_edge->dst(), i);
    //             g->RemoveEdge(o_edge);
    //           }
    //         }
    //       }
    //     }
    //   }
    //   for (auto& opt_ev_name : opt_ev_names) {
    //     Node* opt_node = node_to_origin_map[opt_ev_name][i];
    //     LOG(INFO) << "JUNQI SCALEDOWN BACKWARD VAR===>"
    //               << primary_variable_name << " === " << i
    //               << " === " << opt_node->name();
    //     for (auto* o_node : opt_node->out_nodes()) {
    //       if (o_node->type_string() == kIdentityOp) {
    //         for (auto* o_edge : o_node->out_edges()) {
    //           if (o_edge->dst()->type_string() == "ConcatV2") {
    //             int N;
    //             TF_RETURN_IF_ERROR(GetNodeAttr(o_edge->dst()->attrs(), "N", &N));
    //             if (N != cur_partition_nums_) {
    //               const Edge* axis_edge = nullptr;
    //               TF_RETURN_IF_ERROR(o_edge->dst()->input_edge(N, &axis_edge));
    //               g->AddEdge(axis_edge->src(), 0, o_edge->dst(), cur_partition_nums_);
    //               g->RemoveEdge(axis_edge);
    //               o_edge->dst()->ClearAttr("N");
    //               o_edge->dst()->AddAttr("N", cur_partition_nums_);
    //             }
    //             if (o_edge->dst_input() != i){
    //               g->AddEdge(o_node, 0, o_edge->dst(), i);
    //               g->RemoveEdge(o_edge);
    //             }
    //           }
    //         }
    //       } 
    //     }
    //   }
    // } else {
    //   LOG(INFO) << "JUNQI SCALEDOWN VAR ===>" << primary_variable_name
    //             << " === " << i << " === " << ev_node->name();
    //   for (auto* o_node : ev_node->out_nodes()) {
    //     if (o_node->type_string() == kIdentityOp) {
    //       nodes_to_delete.insert(o_node);
    //       for (auto* oo_node : o_node->out_nodes()) {
    //         if (oo_node->type_string() == "GatherV2") {
    //           nodes_to_delete.insert(oo_node);
    //           var_type = VarType::REF_VAR;
    //           // TODO axis
    //         }
    //       }
    //     } else if (o_node->type_string() == "Assign") {
    //       nodes_to_delete.insert(o_node);
    //     }
    //   }
    //   for (auto& opt_ev_name : opt_ev_names) {
    //     Node* opt_node = node_to_origin_map[opt_ev_name][i];
    //     LOG(INFO) << "JUNQI SCALEDOWN BACKWARD VAR ===>"
    //               << primary_variable_name << " === " << i
    //               << " === " << opt_node->name();
    //     for (auto* o_node : opt_node->out_nodes()) {
    //       if (o_node->type_string() == kIdentityOp) {
    //         nodes_to_delete.insert(o_node);
    //       } else if (o_node->type_string() == "Assign") {
    //         nodes_to_delete.insert(o_node);
    //       }
    //     }
    //   }
    // }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownVarForWardGraph(const VarType& var_type, Graph* g, 
                                                      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                                      std::unordered_set<Node*>& nodes_to_delete,
                                                      const std::string& primary_variable_name,
                                                      const std::vector<std::string>& opt_ev_names,
                                                      int ev_partition_num) {

  for (int i = cur_partition_nums_; i < ev_partition_num; ++i) {
    Node* ev_node = node_to_origin_map[primary_variable_name][i];
    LOG(INFO) << "JUNQI SCALEDOWN VAR ===>" << primary_variable_name
              << " === " << i << " === " << ev_node->name();
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == kIdentityOp) {
        nodes_to_delete.insert(o_node);
        for (auto* oo_node : o_node->out_nodes()) {
          if (oo_node->type_string() == "GatherV2") {
            nodes_to_delete.insert(oo_node);
            // TODO axis
          }
        }
      } else if (o_node->type_string() == "Assign") {
        nodes_to_delete.insert(o_node);
      }
    }
    for (auto& opt_ev_name : opt_ev_names) {
      Node* opt_node = node_to_origin_map[opt_ev_name][i];
      LOG(INFO) << "JUNQI SCALEDOWN BACKWARD VAR ===>"
                << primary_variable_name << " === " << i
                << " === " << opt_node->name();
      for (auto* o_node : opt_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          nodes_to_delete.insert(o_node);
        } else if (o_node->type_string() == "Assign") {
          nodes_to_delete.insert(o_node);
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownForWardGraph(const VarType& var_type, Graph* g, 
                                                    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                                    std::unordered_set<Node*>& nodes_to_delete,
                                                    const std::string& primary_variable_name,
                                                    const std::vector<std::string>& opt_ev_names,
                                                    int ev_partition_num) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      s = ScalingDownEVForWardGraph(
          var_type, g, node_to_origin_map, nodes_to_delete,
          primary_variable_name, opt_ev_names, ev_partition_num);
      break;
    case VarType::RESOURCE_VAR:
      s = ScalingDownResVarForWardGraph(
          var_type, g, node_to_origin_map, nodes_to_delete,
          primary_variable_name, opt_ev_names, ev_partition_num);
      break;
    default:
      s = ScalingDownVarForWardGraph(
          var_type, g, node_to_origin_map, nodes_to_delete,
          primary_variable_name, opt_ev_names, ev_partition_num);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingDownEVForWardGraph(const VarType& var_type, Graph* g, 
                                                    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                                    std::unordered_set<Node*>& nodes_to_delete,
                                                    const std::string& primary_ev_name,
                                                    const std::vector<std::string>& opt_ev_names,
                                                    int ev_partition_num) {
  for (int i = cur_partition_nums_; i < ev_partition_num; ++i) {
    Node* var_node = node_to_origin_map[primary_ev_name][i];
    LOG(INFO) << "JUNQI SCALEDOWN EV ===>" << primary_ev_name
              << " ===== " << i << " === " << var_node->name();
    for (auto* o_node : var_node->out_nodes()) {
      // InitializeEVResource
      if (o_node->type_string() == kEvInitOp) {
        const Node* tmp_check_ev_0;
        TF_RETURN_IF_ERROR(o_node->input_node(0, &tmp_check_ev_0));
        const Node* tmp_check_ev_1;
        TF_RETURN_IF_ERROR(o_node->input_node(1, &tmp_check_ev_1));
        if (tmp_check_ev_0->name() != tmp_check_ev_1->name()) continue;

        nodes_to_delete.insert(o_node);
        const Edge* init_value_edge = nullptr;
        TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
        nodes_to_delete.insert(init_value_edge->src());
        const Edge* empty_key_edge = nullptr;
        TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
        nodes_to_delete.insert(empty_key_edge->src());
      } else if (o_node->type_string() == "KvResourceGather") {
        nodes_to_delete.insert(o_node);
        for (auto* o_edge : o_node->out_edges()) {
          if (o_edge->dst()->type_string() == kIdentityOp) {
            nodes_to_delete.insert(o_edge->dst());
          }
        }
      } else {
        //       nodes_to_delete.insert(o_node);
      }
    }

    for (auto& opt_ev_name : opt_ev_names) {
      Node* ev_node = node_to_origin_map[opt_ev_name][i];
      // nodes_to_delete.insert(ev_node);
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kEvInitOp) {
          nodes_to_delete.insert(o_node);
          const Edge* init_value_edge = nullptr;
          TF_RETURN_IF_ERROR(o_node->input_edge(2, &init_value_edge));
          nodes_to_delete.insert(init_value_edge->src());
          const Edge* empty_key_edge = nullptr;
          TF_RETURN_IF_ERROR(o_node->input_edge(3, &empty_key_edge));
          nodes_to_delete.insert(empty_key_edge->src());
        } else {
          // nodes_to_delete.insert(o_node);
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpBackWardGraph(VarType var_type, Graph* g, 
                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                  const std::string& primary_variable_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  Node* elastic_node, Node* p_dynamic_stitch_node, 
				  std::vector<Node*>& no_op_vec, int part_var_full_shape, int ev_partition_num) {
  Status s;

  switch (var_type) {
    case VarType::EMBEDDING_VAR:
    case VarType::REF_VAR:
    case VarType::RESOURCE_VAR:
      // corresponding to tensorflow.python.training.optimizer.py : _resource_apply_sparse_duplicate_indices
      s = ScalingUpEmbeddingVariableBackWardGraph(
          var_type, g, node_to_origin_map, primary_variable_name, opt_ev_names,
          elastic_node, p_dynamic_stitch_node, no_op_vec, part_var_full_shape, ev_partition_num);
      break;
    default:
      s = ScalingUpDenseBackWardGraph(var_type, g, node_to_origin_map, primary_variable_name, opt_ev_names,
          elastic_node, no_op_vec, part_var_full_shape,
          ev_partition_num);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingUpEmbeddingVariableBackWardGraph(
    VarType var_type, Graph* g, 
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names, Node* elastic_node, Node* p_dynamic_stitch_node,
    std::vector<Node*>& no_op_vec, int part_var_full_shape, int ev_partition_num) {
  LOG(INFO) << "Calling ScalingUpEmbeddingVariableBackWardGraph ---- ";
  Node* ev_node = node_to_origin_map[primary_ev_name][0];
  for (int i = 0; i < cur_partition_nums_; ++i) {
    Node* cur_ev_node = node_to_origin_map[primary_ev_name][i];
    Node* cur_noop_node = no_op_vec[i];
    if ( i < ev_partition_num) {
      for (auto* node : cur_ev_node->out_nodes()) {
        if (IsApplyNode(var_type, node)) {
          for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
            Node* i_node;
            TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
            if (i_node->IsUnique()) {
              Node* unique_indices_node;
              TF_RETURN_IF_ERROR(i_node->input_node(0, &unique_indices_node));
              TF_RETURN_IF_ERROR(g->UpdateEdge(elastic_node, i, unique_indices_node, 0));
              Node* expand_dim_node;
              TF_RETURN_IF_ERROR(unique_indices_node->input_node(1, &expand_dim_node));
              Node* size_node;
              TF_RETURN_IF_ERROR(expand_dim_node->input_node(0, &size_node));
              TF_RETURN_IF_ERROR(g->UpdateEdge(elastic_node, i, size_node, 0));
            }

            if (i_node->type_string() == "UnsortedSegmentSum") {
              Node* reshape_node;
              TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_node));
              Node* control_dependency_node;
              TF_RETURN_IF_ERROR(reshape_node->input_node(0, &control_dependency_node));
              Node* gather_node;
              TF_RETURN_IF_ERROR(control_dependency_node->input_node(0, &gather_node));
              //embedding lookup sparse has extra identity node
              TF_RETURN_IF_ERROR(g->UpdateEdge(elastic_node, cur_partition_nums_ + i, gather_node, 1));
              // Node* gather_input_node;
              // TF_RETURN_IF_ERROR(gather_node->input_node(0, &gather_input_node));
              //BELOW is embedding lookup sparse 's extra SparseSegmentMeanGrad
              // Node* shape_node;
              // TF_RETURN_IF_ERROR(gather_input_node->input_node(1, &shape_node));
              // TF_RETURN_IF_ERROR(g->UpdateEdge(p_dynamic_stitch_node, 0, shape_node, 0));
              // embedding lookup
              // Node* gather_grad_node;
              // TF_RETURN_IF_ERROR(control_dependency_node->input_node(0, &gather_grad_node));
              // TF_RETURN_IF_ERROR(g->UpdateEdge(elastic_node, cur_partition_nums_ + i, gather_grad_node, 1));
            }
          }
        }
      }
    } else {
      string new_device_name = cur_ev_node->assigned_device_name();
      for (auto* node : ev_node->out_nodes()) {
        if (IsApplyNode(var_type, node)) {
          if (cur_noop_node == nullptr) {
            Status s;
            NodeDef noop_def;
            TF_RETURN_IF_ERROR(
                NodeDefBuilder("head/Optimizer/update/NoOp_" + std::to_string(i),
                              "NoOp")
                    .Device(new_device_name)
                    .Finalize(&noop_def));
            Node* no_node = g->AddNode(noop_def, &s);
            no_node ->set_assigned_device_name(new_device_name);
            TF_RETURN_IF_ERROR(s);
            for (auto* edge : node->out_edges()) {
              for (auto* o_edge : edge->dst()->out_edges()) {
                if (o_edge->IsControlEdge()) {
                  g->AddControlEdge(no_node, o_edge->dst());
                }
              }
            }
            cur_noop_node = no_node;
            no_op_vec[i] = no_node;
          }

          LOG(INFO) << "Copying new apply node";
          Node* new_apply_node =
              CopyNode(g, node, new_device_name, i);
          g->AddControlEdge(new_apply_node, cur_noop_node);
          g->AddEdge(node_to_origin_map[primary_ev_name][i], 0, new_apply_node,
                    0);
          for (int j = 0; j < opt_ev_names.size(); ++j) {
            g->AddEdge(node_to_origin_map[opt_ev_names[j]][i], 0, new_apply_node,
                      j + 1);
          }
          Node* new_unique;
          Node* new_expand_dims;
          for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
            Node* i_node;
            TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
            if (i_node->IsUnique()) {
              LOG(INFO) << "Copying new unique node";
              new_unique =
                  CopyNode(g, i_node, new_device_name, i);
              g->AddEdge(new_unique, 0, new_apply_node, j);
              // unique INPUT 0
              LOG(INFO) << "Copying reshape of unique input";
              Node* reshape_id;
              TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_id));
              Node* new_reshape_id =
                  CopyNode(g, reshape_id, reshape_id->assigned_device_name(), i);
              g->AddEdge(new_reshape_id, 0, new_unique, 0);

              for (auto* o_node : reshape_id->out_nodes()) {
                if (o_node->type_string() == "RecordSparseIndices") {
                  Node* new_record_sparse = CopyNode(
                      g, o_node, new_device_name, i);
                  g->AddEdge(new_reshape_id, 0, new_record_sparse, 0);
                  g->AddControlEdge(new_record_sparse, cur_noop_node);
                }
              }

              // Reshape INPUT
              g->AddEdge(elastic_node, i, new_reshape_id, 0);

              Node* expand_dims;
              TF_RETURN_IF_ERROR(reshape_id->input_node(1, &expand_dims));
              new_expand_dims =
                  CopyNode(g, expand_dims, expand_dims->assigned_device_name(), i);
              g->AddEdge(new_expand_dims, 0, new_reshape_id, 1);

              // expand dims INPUT
              Node* expand_dims_size;
              TF_RETURN_IF_ERROR(expand_dims->input_node(0, &expand_dims_size));
              Node* new_expand_dims_size = CopyNode(
                  g, expand_dims_size, expand_dims_size->assigned_device_name(), i);
              g->AddEdge(new_expand_dims_size, 0, new_expand_dims, 0);
              g->AddEdge(elastic_node, i, new_expand_dims_size, 0);

              Node* expand_dims_dim;
              TF_RETURN_IF_ERROR(expand_dims->input_node(1, &expand_dims_dim));
              Node* new_expand_dims_dim = CopyNode(
                  g, expand_dims_dim, expand_dims_dim->assigned_device_name(), i);
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
              Node* new_unsorted_segment = CopyNode(
                  g, i_node, new_device_name, i);
              g->AddEdge(new_unsorted_segment, 0, new_apply_node, j);
              // Input 0
              {
                Node* reshape;
                TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape));
                Node* new_reshape =
                    CopyNode(g, reshape, reshape->assigned_device_name(), i);
                g->AddEdge(new_reshape, 0, new_unsorted_segment, 0);
                // Reshape INPUT 0
                Node* control_denpency;
                TF_RETURN_IF_ERROR(reshape->input_node(0, &control_denpency));
                Node* new_control_denpency = CopyNode(
                    g, control_denpency, control_denpency->assigned_device_name(), i);
                g->AddEdge(new_control_denpency, 0, new_reshape, 0);

                for (auto* i_edge : control_denpency->in_edges()) {
                  if (i_edge->IsControlEdge()) {
                    g->AddControlEdge(i_edge->src(), new_control_denpency);
                  }
                }

                // control_dependency INPUT 0
                Node* gather_1;
                TF_RETURN_IF_ERROR(control_denpency->input_node(0, &gather_1));
                Node* new_gather_1 =
                    CopyNode(g, gather_1, gather_1->assigned_device_name(), i);
                g->AddEdge(new_gather_1, 0, new_control_denpency, 0);
                for (auto* o_edge : gather_1->out_edges()) {
                  if (o_edge->IsControlEdge()) {
                    g->AddControlEdge(new_gather_1, o_edge->dst());
                  }
                }

                Node* reshape_1;
                TF_RETURN_IF_ERROR(gather_1->input_node(0, &reshape_1));
                g->AddEdge(reshape_1, 0, new_gather_1, 0);

                // gather_1 INPUT1
                g->AddEdge(elastic_node, cur_partition_nums_ + i /*idx*/,
                          new_gather_1, 1);
                // gather_1 INPUT2
                Node* axis_1;
                TF_RETURN_IF_ERROR(gather_1->input_node(2, &axis_1));
                Node* new_axis_1 =
                    CopyNode(g, axis_1, axis_1->assigned_device_name(), i);
                g->AddEdge(new_axis_1, 0, new_gather_1, 2);

                // Reshape INPUT 1
                Node* concat;
                TF_RETURN_IF_ERROR(reshape->input_node(1, &concat));
                Node* new_concat =
                    CopyNode(g, concat, concat->assigned_device_name(), i);
                g->AddEdge(new_concat, 0, new_reshape, 1);

                // concat INPUT 0
                g->AddEdge(new_expand_dims, 0, new_concat, 0);

                // concat INPUT 1
                Node* strided_slice;
                TF_RETURN_IF_ERROR(concat->input_node(1, &strided_slice));
                Node* new_strided_slice = CopyNode(
                    g, strided_slice, strided_slice->assigned_device_name(), i);
                g->AddEdge(new_strided_slice, 0, new_concat, 1); // Const shape

                for (int k = 0; k < strided_slice->num_inputs(); ++k) {
                  Node* partial_strided_slice;
                  TF_RETURN_IF_ERROR(
                      strided_slice->input_node(k, &partial_strided_slice));
                  Node* new_node =
                      CopyNode(g, partial_strided_slice,
                              partial_strided_slice->assigned_device_name(), i);
                  g->AddEdge(new_node, 0, new_strided_slice, k);
                }

                // concat INPUT 2
                Node* axis;
                TF_RETURN_IF_ERROR(concat->input_node(2, &axis));
                Node* new_axis = CopyNode(g, axis, axis->assigned_device_name(), i);
                g->AddEdge(new_axis, 0, new_concat, 2);
              }

              // Input 1
              g->AddEdge(new_unique, 1 /*idx*/, new_unsorted_segment, 1);
              LOG(INFO) << "Copying  UnsortedSegmentSum node INPUT 2";
              // Input 2
              {
                Node* strided_slice;
                TF_RETURN_IF_ERROR(i_node->input_node(2, &strided_slice));
                Node* new_strided_slice =
                    CopyNode(g, strided_slice, new_device_name, i);
                g->AddEdge(new_strided_slice, 0, new_unsorted_segment, 2);

                Node* shape;
                TF_RETURN_IF_ERROR(strided_slice->input_node(0, &shape));
                Node* new_shape = CopyNode(g, shape, new_device_name, i);
                g->AddEdge(new_unique, 0, new_shape, 0);
                g->AddEdge(new_shape, 0, new_strided_slice, 0);

                for (int k = 1; k < strided_slice->num_inputs(); ++k) {
                  Node* partial_strided_slice;
                  TF_RETURN_IF_ERROR(
                      strided_slice->input_node(k, &partial_strided_slice));
                  Node* new_node =
                      CopyNode(g, partial_strided_slice, new_device_name, i);
                  g->AddEdge(new_node, 0, new_strided_slice, k);
                }
              }
            } else {
              g->AddEdge(i_node, 0, new_apply_node, j);
            }
          }
        }
      }
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownDenseBackWardGraph(
    Graph* g,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete,
    const std::string& primary_ev_name,
    Node* elastic_node, int part_var_full_shape,
    int ev_partition_num) {
  LOG(INFO) << "calling ScalingDownDenseBackWardGraph";
  for (int i = 0; i < ev_partition_num; ++i) {
    Node* cur_var_node = node_to_origin_map[primary_ev_name][i];
    if ( i < cur_partition_nums_) {
      for (auto* node : cur_var_node->out_nodes()) {
        if (node->IsApplyAdamOps() || node->IsApplyAdagradOps() ||
            node->IsApplyFtrlOps()) {
          Node* i_node;
          TF_RETURN_IF_ERROR(node->input_node(node->num_inputs() - 1, &i_node));
          if (i_node->type_string() == "Identity") {
            Node* concat_grad_node;
            TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
            Node* shape_node;
            TF_RETURN_IF_ERROR(concat_grad_node->input_node(2, &shape_node));
            Tensor old_shape_tensor;
            TF_RETURN_IF_ERROR(GetNodeAttr(shape_node->attrs(), "value", &old_shape_tensor));
            int tensor_size = old_shape_tensor.NumElements();
            int new_part_shape;
            if (i != cur_partition_nums_ - 1) {
              new_part_shape = part_var_full_shape / cur_partition_nums_;
              LOG(INFO) << "new grad shape" << part_var_full_shape << "new is " << new_part_shape;
            } else {
              new_part_shape = part_var_full_shape - part_var_full_shape / cur_partition_nums_ * (cur_partition_nums_ - 1);
              LOG(INFO) << "new grad shape" << part_var_full_shape << "new is " << new_part_shape;
            }
            
            Tensor shape_tensor;
            TensorProto tensor_shape_proto;
            tensor_shape_proto.set_dtype(DT_INT32);
            TensorShape({tensor_size})
                .AsProto(tensor_shape_proto.mutable_tensor_shape());
            tensor_shape_proto.add_int_val(new_part_shape);
            if (tensor_size > 1) {
              for (int j = 1; j < tensor_size;  ++j) {
                tensor_shape_proto.add_int_val(old_shape_tensor.flat<int>()(j));
              }
            }
            shape_tensor.FromProto(tensor_shape_proto);
            shape_node->ClearAttr("value");
            shape_node->AddAttr("value", shape_tensor);

            const Edge* concat_offset_edge;
            TF_RETURN_IF_ERROR(concat_grad_node->input_edge(1, &concat_offset_edge));
            const Edge* target_edge =nullptr;
            for (auto *o_edge: shape_node->out_edges()) {
              if (o_edge->dst() == concat_offset_edge->src()) {
                target_edge = o_edge;
              }
            }
            g->RemoveEdge(target_edge);
            g->AddEdge(shape_node, 0, concat_offset_edge->src(), i+1);
            //concat offset grad
            if (concat_offset_edge->src_output() != i) {
              g->UpdateEdge(concat_offset_edge->src(), i, concat_grad_node, 1);
            }
          }
        }
      }
    } else {
      for (auto* node : cur_var_node->out_nodes()) {
        if (node->IsApplyAdamOps() || node->IsApplyAdagradOps() ||
            node->IsApplyFtrlOps()) {
          nodes_to_delete.insert(node);

          Node* i_node;
          TF_RETURN_IF_ERROR(node->input_node(node->num_inputs() - 1, &i_node));
          if (i_node->type_string() == "Identity") {
            nodes_to_delete.insert(i_node);

            Node* concat_grad_node;
            TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
            nodes_to_delete.insert(concat_grad_node);

            Node* prev_grad_node;
            TF_RETURN_IF_ERROR(
                concat_grad_node->input_node(0, &prev_grad_node));

            Node* concat_offset_node;
            TF_RETURN_IF_ERROR(
                concat_grad_node->input_node(1, &concat_offset_node));
            int part_num;
            TF_RETURN_IF_ERROR(
                GetNodeAttr(concat_offset_node->attrs(), "N", &part_num));
            Node* shape_node;
            TF_RETURN_IF_ERROR(
                concat_grad_node->input_node(2, &shape_node));
            nodes_to_delete.insert(shape_node);

            if (i == ev_partition_num-1) {
              concat_offset_node->ClearAttr("N");
              concat_offset_node->AddAttr("N", cur_partition_nums_);
            }
          } 
        }
      }
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownBackWardGraph(
    Graph* g, VarType var_type,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete,
    const std::string& primary_variable_name,
    const std::vector<std::string>& opt_ev_names, 
    Node* elastic_node, Node* p_dynamic_stitch_node, 
    int part_var_full_shape, int ev_partition_num) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
    case VarType::REF_VAR:
    case VarType::RESOURCE_VAR:
      s = ScalingDownEmbeddingVariableBackWardGraph(
          var_type, g, node_to_origin_map, nodes_to_delete,
          primary_variable_name, opt_ev_names, 
          elastic_node, p_dynamic_stitch_node, part_var_full_shape, ev_partition_num);
      break;
    default:
      s = ScalingDownDenseBackWardGraph(
          g, node_to_origin_map, nodes_to_delete,
          primary_variable_name, 
          elastic_node, part_var_full_shape, ev_partition_num);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingDownEmbeddingVariableBackWardGraph(
    VarType var_type, Graph* g,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_set<Node*>& nodes_to_delete,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names, 
    Node* elastic_node, Node* p_dynamic_stitch_node,
    int part_var_full_shape, int ev_partition_num) {

  for (int i = 0; i < ev_partition_num; ++i) {
    Node* cur_ev_node = node_to_origin_map[primary_ev_name][i];
    if ( i < cur_partition_nums_) {
      for (auto* node : cur_ev_node->out_nodes()) {
        if (IsApplyNode(var_type, node)) {
          for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
            Node* i_node;
            TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
            /*
              ElasticPartition: ID -> Reshape -> Unique: ids ->   Unique
                        expand_dim ->

            */
            if (i_node->IsUnique()) {
              Node* unique_indices_node;
              TF_RETURN_IF_ERROR(i_node->input_node(0, &unique_indices_node));
              TF_RETURN_IF_ERROR(g->UpdateEdge(elastic_node, i, unique_indices_node, 0));
              Node* expand_dim_node;
              TF_RETURN_IF_ERROR(unique_indices_node->input_node(1, &expand_dim_node));
              Node* size_node;
              TF_RETURN_IF_ERROR(expand_dim_node->input_node(0, &size_node));
              TF_RETURN_IF_ERROR(g->UpdateEdge(elastic_node, i, size_node, 0));
            }

            if (i_node->type_string() == "UnsortedSegmentSum") {
              Node* reshape_node;
              TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_node));
              Node* control_dependency_node;
              TF_RETURN_IF_ERROR(reshape_node->input_node(0, &control_dependency_node));
              Node* gather_node;
              TF_RETURN_IF_ERROR(control_dependency_node->input_node(0, &gather_node));
              TF_RETURN_IF_ERROR(g->UpdateEdge(elastic_node, cur_partition_nums_ + i, gather_node, 1));
              // Node* gather_input_node;
              // TF_RETURN_IF_ERROR(gather_node->input_node(0, &gather_input_node));
              //BELOW is embedding lookup sparse 's extra SparseSegmentMeanGrad
              // Node* shape_node;
              // TF_RETURN_IF_ERROR(gather_input_node->input_node(1, &shape_node));
              // TF_RETURN_IF_ERROR(g->UpdateEdge(p_dynamic_stitch_node, 0, shape_node, 0));

              // Node* control_dependency_input_node;
              // TF_RETURN_IF_ERROR(gather_input_node->input_node(0, &control_dependency_input_node));
              /*
              Node* gather_control_dep_node;
              TF_RETURN_IF_ERROR(gather_input_node->input_node(0, &gather_control_dep_node));
              Node* slice_grad_node;
              TF_RETURN_IF_ERROR(gather_control_dep_node->input_node(0, &slice_grad_node));

              const Edge* slice_grad_edge;
              TF_RETURN_IF_ERROR(slice_grad_node->input_edge(1, &slice_grad_edge));
              const Edge* target_edge =nullptr;

              if (slice_grad_edge->src_output() != i) {
                TF_RETURN_IF_ERROR(g->UpdateEdge(slice_grad_edge->src(), i, slice_grad_node, 1));
              }*/
            }
          }
        }
      }
    } else {
      for (auto* node : cur_ev_node->out_nodes()) {
        if (IsApplyNode(var_type, node)) {
          nodes_to_delete.insert(node);
          for (int j = node->num_inputs() - 1; j > opt_ev_names.size(); --j) {
            Node* i_node;
            TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
            if (i_node->IsUnique()) {
              nodes_to_delete.insert(i_node);
              // unique INPUT 0
              Node* reshape_id;
              TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape_id));
              nodes_to_delete.insert(reshape_id);

              for (auto* o_node : reshape_id->out_nodes()) {
                if (o_node->type_string() == "RecordSparseIndices") {
                  nodes_to_delete.insert(o_node);
                }
              }

              Node* expand_dims;
              TF_RETURN_IF_ERROR(reshape_id->input_node(1, &expand_dims));
              nodes_to_delete.insert(expand_dims);

              // expand dims INPUT
              Node* expand_dims_size;
              TF_RETURN_IF_ERROR(expand_dims->input_node(0, &expand_dims_size));
              nodes_to_delete.insert(expand_dims_size);

              Node* expand_dims_dim;
              TF_RETURN_IF_ERROR(expand_dims->input_node(1, &expand_dims_dim));
              nodes_to_delete.insert(expand_dims_dim);

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
              nodes_to_delete.insert(i_node);
              // Input 0
              {
                Node* reshape;
                TF_RETURN_IF_ERROR(i_node->input_node(0, &reshape));
                nodes_to_delete.insert(reshape);
                // Reshape INPUT 0
                Node* control_denpency;
                TF_RETURN_IF_ERROR(reshape->input_node(0, &control_denpency));
                nodes_to_delete.insert(control_denpency);

                // control_dependency INPUT 0
                Node* gather_1;
                TF_RETURN_IF_ERROR(control_denpency->input_node(0, &gather_1));
                nodes_to_delete.insert(gather_1);

                // gather_1 INPUT2
                Node* axis_1;
                TF_RETURN_IF_ERROR(gather_1->input_node(2, &axis_1));
                nodes_to_delete.insert(axis_1);

                // Reshape INPUT 1
                Node* concat;
                TF_RETURN_IF_ERROR(reshape->input_node(1, &concat));
                nodes_to_delete.insert(concat);

                // concat INPUT 1
                Node* strided_slice;
                TF_RETURN_IF_ERROR(concat->input_node(1, &strided_slice));
                nodes_to_delete.insert(strided_slice);

                for (int k = 0; k < strided_slice->num_inputs(); ++k) {
                  Node* partial_strided_slice;
                  TF_RETURN_IF_ERROR(
                      strided_slice->input_node(k, &partial_strided_slice));
                  nodes_to_delete.insert(partial_strided_slice);
                }

                // concat INPUT 2
                Node* axis;
                TF_RETURN_IF_ERROR(concat->input_node(2, &axis));
                nodes_to_delete.insert(axis);
              }

              // Input 2
              {
                Node* strided_slice;
                TF_RETURN_IF_ERROR(i_node->input_node(2, &strided_slice));
                nodes_to_delete.insert(strided_slice);

                Node* shape;
                TF_RETURN_IF_ERROR(strided_slice->input_node(0, &shape));
                nodes_to_delete.insert(shape);

                for (int k = 1; k < strided_slice->num_inputs(); ++k) {
                  Node* partial_strided_slice;
                  TF_RETURN_IF_ERROR(
                      strided_slice->input_node(k, &partial_strided_slice));
                  nodes_to_delete.insert(partial_strided_slice);
                }
              }
            }
          }
        }
      }
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpDenseBackWardGraph(
    VarType var_type, Graph* g,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    const std::string& primary_ev_name,
    const std::vector<std::string>& opt_ev_names, 
    Node* elastic_node, std::vector<Node*>& no_op_vec,
    int part_var_full_shape, int ev_partition_num) {
  LOG(INFO) << "calling ScalingUpDenseBackWardGraph";
  Node* var_node = node_to_origin_map[primary_ev_name][0];
  for (int i = 0; i < cur_partition_nums_; ++i) {
    Node* cur_var_node = node_to_origin_map[primary_ev_name][i];
    if ( i < ev_partition_num) {
      for (auto* node : cur_var_node->out_nodes()) {
        if (node->IsApplyAdamOps() || node->IsApplyAdagradOps() ||
            node->IsApplyFtrlOps()) {
          Node* i_node;
          TF_RETURN_IF_ERROR(node->input_node(node->num_inputs() - 1, &i_node));
          if (i_node->type_string() == "Identity") {
            Node* concat_grad_node;
            TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
            Node* shape_node;
            TF_RETURN_IF_ERROR(concat_grad_node->input_node(2, &shape_node));
            Tensor old_shape_tensor;
            TF_RETURN_IF_ERROR(GetNodeAttr(shape_node->attrs(), "value", &old_shape_tensor));
            int tensor_size = old_shape_tensor.NumElements();
            int new_part_shape = part_var_full_shape / cur_partition_nums_;
            Tensor shape_tensor;
            TensorProto tensor_shape_proto;
            tensor_shape_proto.set_dtype(DT_INT32);
            TensorShape({tensor_size})
                .AsProto(tensor_shape_proto.mutable_tensor_shape());
            tensor_shape_proto.add_int_val(new_part_shape);
            if (tensor_size > 1) {
              for (int j = 1; j < tensor_size;  ++j) {
                tensor_shape_proto.add_int_val(old_shape_tensor.flat<int>()(j));
              }
            }
            shape_tensor.FromProto(tensor_shape_proto);
            shape_node->ClearAttr("value");
            shape_node->AddAttr("value", shape_tensor);
          }
        }
      }
    }
    else {
      Node* cur_noop_node = no_op_vec[i];
      string new_device_name = cur_var_node->assigned_device_name();
      for (auto* node : var_node->out_nodes()) {
        if (node->IsApplyAdamOps() || node->IsApplyAdagradOps() ||
            node->IsApplyFtrlOps()) {
          if (cur_noop_node == nullptr) {
            Status s;
            NodeDef noop_def;
            TF_RETURN_IF_ERROR(
                NodeDefBuilder("head/Optimizer/update/NoOp_" + std::to_string(i),
                              "NoOp")
                    .Device(new_device_name)
                    .Finalize(&noop_def));
            Node* no_node = g->AddNode(noop_def, &s);
            TF_RETURN_IF_ERROR(s);
            for (auto* edge : node->out_edges()) {
              for (auto* o_edge : edge->dst()->out_edges()) {
                if (o_edge->IsControlEdge()) {
                  g->AddControlEdge(no_node, o_edge->dst());
                }
              }
            }
            cur_noop_node = no_node;
            no_op_vec[i] = no_node;
          }

          Node* new_apply_node =
              CopyNode(g, node, new_device_name, i);
          g->AddControlEdge(new_apply_node, cur_noop_node);
          g->AddEdge(node_to_origin_map[primary_ev_name][i], 0, new_apply_node,
                    0);
          for (int j = 0; j < opt_ev_names.size(); ++j) {
            g->AddEdge(node_to_origin_map[opt_ev_names[j]][i], 0, new_apply_node,
                      j + 1);
          }

          Node* i_node;
          TF_RETURN_IF_ERROR(node->input_node(node->num_inputs() - 1, &i_node));
          if (i_node->type_string() == "Identity") {
            Node* new_grad_node = CopyNode(g, i_node, new_device_name, i);
            g->AddEdge(new_grad_node, 0, new_apply_node, node->num_inputs() - 1);

            Node* concat_grad_node;
            TF_RETURN_IF_ERROR(i_node->input_node(0, &concat_grad_node));
            Node* new_concat_grad_node = CopyNode(
                g, concat_grad_node, concat_grad_node->assigned_device_name(), i);
            g->AddEdge(new_concat_grad_node, 0, new_grad_node, 0);

            Node* prev_grad_node;
            TF_RETURN_IF_ERROR(concat_grad_node->input_node(0, &prev_grad_node));
            g->AddEdge(prev_grad_node, 0, new_concat_grad_node, 0);

            Node* concat_offset_node;
            TF_RETURN_IF_ERROR(
                concat_grad_node->input_node(1, &concat_offset_node));
            int part_num;
            TF_RETURN_IF_ERROR(
                GetNodeAttr(concat_offset_node->attrs(), "N", &part_num));
          
            if (part_num != cur_partition_nums_) {
              concat_offset_node->ClearAttr("N");
              concat_offset_node->AddAttr("N", cur_partition_nums_);
            }
            
            g->AddEdge(concat_offset_node, i, new_concat_grad_node, 1);
            Node* shape_node;
            TF_RETURN_IF_ERROR(concat_offset_node->input_node(1, &shape_node));
            Tensor old_shape_tensor;
            TF_RETURN_IF_ERROR(GetNodeAttr(shape_node->attrs(), "value", &old_shape_tensor));
            int tensor_size = old_shape_tensor.NumElements();
            Node* new_shape_node =
                CopyNode(g, shape_node, shape_node->assigned_device_name(), i);
            if (i == cur_partition_nums_-1) {
              int new_part_shape = part_var_full_shape - part_var_full_shape / cur_partition_nums_ * (cur_partition_nums_-1);
              Tensor shape_tensor;
              TensorProto tensor_shape_proto;
              tensor_shape_proto.set_dtype(DT_INT32);
              TensorShape({tensor_size})
                  .AsProto(tensor_shape_proto.mutable_tensor_shape());
              tensor_shape_proto.add_int_val(new_part_shape);
              if (tensor_size > 1) {
                for (int j = 1; j < tensor_size;  ++j) {
                  tensor_shape_proto.add_int_val(old_shape_tensor.flat<int>()(j));
                }
              }
              bool ret = shape_tensor.FromProto(tensor_shape_proto);
              if (!ret) return errors::Internal("shape tensor init error");
              new_shape_node->ClearAttr("value");
              new_shape_node->AddAttr("value", shape_tensor);
            }
            g->AddEdge(new_shape_node, 0, concat_offset_node, i+1);
            g->AddEdge(new_shape_node, 0, new_concat_grad_node, 2);
            //TODO grad value size
            for (auto* i_edge : i_node->in_edges()) {
              if (i_edge->IsControlEdge()) {
                Node* control_node = i_edge->src();
                g->AddControlEdge(new_concat_grad_node, control_node);
                g->AddControlEdge(control_node, new_grad_node);
              }
            }
          }

          for (int j = node->num_inputs() - 2; j > opt_ev_names.size(); --j) {
            Node* i_node;
            TF_RETURN_IF_ERROR(node->input_node(j, &i_node));
            g->AddEdge(i_node, 0, new_apply_node, j);
          }
        }
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingUpRedistributionGraph(VarType var_type, Graph* g, std::vector<Node*>& var_vec, Node* import_op_main,
                                                          int ev_partition_num, std::vector<Node*>& primary_ev_filters) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      s = ScalingUpEVRedistributionGraph(var_type, g, var_vec, import_op_main, ev_partition_num, primary_ev_filters);
      break;
    case VarType::RESOURCE_VAR:
    case VarType::DENSE_RESOUCE_VAR:
      s = ScalingUpResVarRedistributionGraph(var_type, g, var_vec, import_op_main, ev_partition_num, primary_ev_filters);
      break;
    default:
      s = ScalingUpVarRedistributionGraph(var_type, g, var_vec, import_op_main, ev_partition_num, primary_ev_filters);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingUpEVRedistributionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& ev_node_vec, Node* import_op_main,
    int ev_partition_num, std::vector<Node*>& primary_ev_filters) {
  LOG(INFO) << "CALLING ScalingUpEVRedistributionGraph ";
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  filtered_node_vec.reserve(cur_partition_nums_);
  for (int i = 0; i < cur_partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    auto* primary_ev_filter_node = primary_ev_filters[i];
    if (i < ev_partition_num) {
      TF_RETURN_IF_ERROR(FindNode(ev_node, kEvExportOp,
          [this, &g, &filtered_node_vec, &primary_ev_filters, &primary_ev_filter_node,
              &key_type, &value_type, i](Node* target_node) {
            TF_RETURN_IF_ERROR(GetNodeAttr(target_node->attrs(), "Tkeys", &key_type));
            TF_RETURN_IF_ERROR(
                GetNodeAttr(target_node->attrs(), "dtype", &value_type));
            filtered_node_vec.push_back(target_node);
            if (primary_ev_filter_node == nullptr) {
              primary_ev_filters[i] = target_node;
            } else {
              g->AddControlEdge(target_node, primary_ev_filters[i]);
            }
            return Status::OK();
          }));
    } else {
      NodeDef filter_storage_node_def;
      TF_RETURN_IF_ERROR(
          NodeDefBuilder("FilterStorage/"+ev_node->name(), kEvExportOp)
              .Input(ev_node->name(), 0, ev_node->output_type(0))
              .Input("partition_num", 0, DT_INT32)
              .Attr("partition_id", i)
              .Attr("Tkeys", key_type)
              .Attr("dtype", value_type)
              .Finalize(&filter_storage_node_def));
      Node* filter_node = g->AddNode(filter_storage_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      filter_node->set_assigned_device_name(ev_node->assigned_device_name());
      filtered_node_vec.push_back(filter_node);
      if (primary_ev_filter_node == nullptr) {
        primary_ev_filters[i] = filter_node;
      } else {
        g->AddControlEdge(filter_node, primary_ev_filters[i]);
      }
    }
  }

  for (int i = 0; i < cur_partition_nums_; ++i) {
    auto* ev_node = ev_node_vec[i];
    std::vector<Node*> sorted_filted_vec {filtered_node_vec[i]};
    for (int j = 0; j < filtered_node_vec.size(); ++j) {
      if (i != j){
        sorted_filted_vec.emplace_back(filtered_node_vec[j]);
      }
    }

    if (i < ev_partition_num) {
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kEvImportOp) {
          o_node->ClearAttr("partition_nums");
          o_node->AddAttr("partition_nums", cur_partition_nums_);
          o_node->set_assigned_device_name(ev_node->assigned_device_name());
          std::vector<const Edge*> in_edges;
          in_edges.reserve(o_node->in_edges().size());
          for (auto* o_edge : o_node->in_edges()) {
            in_edges.emplace_back(o_edge);
          }
          for (const Edge* e : in_edges) {
            g->RemoveEdge(e);
          }
          for (int j = 0; j < sorted_filted_vec.size(); ++j) {
            g->AddEdge(sorted_filted_vec[j], 0, o_node, 1+j);
            g->AddEdge(sorted_filted_vec[j], 1, o_node,
                        1 + cur_partition_nums_ + j);
            g->AddEdge(sorted_filted_vec[j], 2, o_node,
                        1 + cur_partition_nums_ * 2 + j);
            g->AddEdge(sorted_filted_vec[j], 3, o_node,
                        1 + cur_partition_nums_ * 3 + j);
          }
        }
      }
    } else {
      std::vector<NodeDefBuilder::NodeOut> import_keys;
      std::vector<NodeDefBuilder::NodeOut> import_values;
      std::vector<NodeDefBuilder::NodeOut> import_versions;
      std::vector<NodeDefBuilder::NodeOut> import_freqs;
      for (int j = 0; j < sorted_filted_vec.size(); ++j) {
        import_keys.emplace_back(sorted_filted_vec[j]->name(), 0,
                                  sorted_filted_vec[j]->output_type(0));
        import_values.emplace_back(sorted_filted_vec[j]->name(), 1,
                                    sorted_filted_vec[j]->output_type(1));
        import_versions.emplace_back(sorted_filted_vec[j]->name(), 2,
                                      sorted_filted_vec[j]->output_type(2));
        import_freqs.emplace_back(sorted_filted_vec[j]->name(), 3,
                                  sorted_filted_vec[j]->output_type(3));
      }
      NodeDef import_storage_node_def;
      LOG(INFO) << "filter input size: " << import_keys.size()
                << " filtered_node_vec size: " << filtered_node_vec.size()
                << " arrt is: " << cur_partition_nums_;
      TF_RETURN_IF_ERROR(NodeDefBuilder(kEvImportOp+ev_node->name(), kEvImportOp)
                             .Input(ev_node->name(), 0, ev_node->output_type(0))
                             .Input(import_keys)
                             .Input(import_values)
                             .Input(import_versions)
                             .Input(import_freqs)
                             .Attr("partition_id", i)
                             .Attr("partition_nums", cur_partition_nums_)
                             .Attr("Tkeys", key_type)
                             .Attr("dtype", value_type)
                             .Finalize(&import_storage_node_def));
      Node* import_node = g->AddNode(import_storage_node_def, &s);
      TF_RETURN_IF_ERROR(s);
      import_node->set_assigned_device_name(ev_node->assigned_device_name());

      g->AddControlEdge(import_node, import_op_main);
      for (int k = 0; k < ev_partition_num; ++k) {
        auto* tmp_ev_node = ev_node_vec[k];
        for (auto* n : tmp_ev_node->out_nodes()) {
          if (n->type_string() == kEvImportOp) {
            g->AddControlEdge(import_node, n);
          }
        }
      }
    }
  }

  return s;
}

Status ElasticTrainingPass::ScalingUpResVarRedistributionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& var_node_vec,
    Node* import_op_main, int ev_partition_num, std::vector<Node*>& primary_ev_filters) {
  LOG(INFO) << "CALLING ScalingUpResVarRedistributionGraph";
  Status s;
  Node* ori_var = var_node_vec[0];
  Node* rhs_value_node = nullptr;
  Node* partition_num_node = nullptr;
  int partition_num;
  DataType key_type;
  for (auto* oo_node : ori_var->out_nodes()) {
    if (oo_node->type_string() == "ReAssignResource") {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "partition_nums", &partition_num));
      TF_RETURN_IF_ERROR(oo_node->input_node(1, &rhs_value_node));
      TF_RETURN_IF_ERROR(oo_node->input_node(2, &partition_num_node));
      TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "T", &key_type));
    }
  }

  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    auto* var_node = var_node_vec[i];
    NodeDef reassign_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(var_node->name() + "/ReAssignResource", "ReAssignResource")
            .Input(var_node->name(), 0, DT_RESOURCE)
            .Input(rhs_value_node->name(), 0, rhs_value_node->output_type(0))
            .Input(partition_num_node->name(), 0,
                    partition_num_node->output_type(0))
            .Attr("partition_id", i)
            .Attr("partition_nums", partition_num)
            .Attr("T", key_type)
            .Finalize(&reassign_node_def));
    Node* reassign_node = g->AddNode(reassign_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    reassign_node->set_assigned_device_name(var_node->assigned_device_name());
    g->AddControlEdge(reassign_node, import_op_main);
  }
  return s;
}

Status ElasticTrainingPass::ScalingUpVarRedistributionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& var_node_vec, Node* import_op_main,
    int ev_partition_num, std::vector<Node*>& primary_ev_filters) {
  LOG(INFO) << "CALLING ScalingUpVarRedistributionGraph";
  Status s;
  Node* ori_var = var_node_vec[0];
  bool use_locking;
  int partition_num;
  DataType key_type;
  Node* rhs_value_node = nullptr;
  Node* partition_num_node = nullptr;
  for (auto* oo_node : ori_var->out_nodes()) {
    if (oo_node->type_string() == "ReAssign") {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "use_locking", &use_locking));
      TF_RETURN_IF_ERROR(
          GetNodeAttr(oo_node->attrs(), "partition_nums", &partition_num));
      TF_RETURN_IF_ERROR(GetNodeAttr(oo_node->attrs(), "T", &key_type));
      TF_RETURN_IF_ERROR(oo_node->input_node(1, &rhs_value_node));
      TF_RETURN_IF_ERROR(oo_node->input_node(2, &partition_num_node));
    }
  }

  for (int i = ev_partition_num; i < cur_partition_nums_; ++i) {
    auto* var_node = var_node_vec[i];
    NodeDef reassign_node_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(var_node->name() + "/ReAssign", "ReAssign")
            .Input(var_node->name(), 0, MakeRefType(key_type))
            .Input(rhs_value_node->name(), 0, rhs_value_node->output_type(0))
            .Input(partition_num_node->name(), 0,
                    partition_num_node->output_type(0))
            .Attr("use_locking", use_locking)
            .Attr("partition_id", i)
            .Attr("partition_nums", partition_num)
            .Attr("T", key_type)
            .Finalize(&reassign_node_def));
    Node* reassign_node = g->AddNode(reassign_node_def, &s);
    TF_RETURN_IF_ERROR(s);
    reassign_node->set_assigned_device_name(var_node->assigned_device_name());
    g->AddControlEdge(reassign_node, import_op_main);
  }
  return s;
}

Status ElasticTrainingPass::ScalingDownResVarRedistributionGraph(
    Graph* g, std::vector<Node*>& ev_node_vec, 
    std::unordered_set<Node*>& nodes_to_delete,
    int ev_partition_num) {
  LOG(INFO) << "CALLING ScalingDownResVarRedistributionGraph";

  for (int i = cur_partition_nums_; i < ev_node_vec.size(); ++i) {
    auto* ev_node = ev_node_vec[i];
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == "ReAssignResource") {
        nodes_to_delete.emplace(o_node);
      }
    }
  }
  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownVarRedistributionGraph(
    Graph* g, std::vector<Node*>& ev_node_vec, 
    std::unordered_set<Node*>& nodes_to_delete,
    int ev_partition_num) {
  LOG(INFO) << "CALLING ScalingDownVarRedistributionGraph";

  for (int i = 0; i < ev_partition_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    if (i < cur_partition_nums_) {
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          for (auto* o_edge : o_node->out_edges()) {
            // Normal Variable
            if (o_edge->dst()->type_string() == "ConcatV2") {
              int N;
              TF_RETURN_IF_ERROR(GetNodeAttr(o_edge->dst()->attrs(), "N", &N));
              if (N != cur_partition_nums_) {
                const Edge* axis_edge = nullptr;
                TF_RETURN_IF_ERROR(o_edge->dst()->input_edge(N, &axis_edge));
                g->AddEdge(axis_edge->src(), 0, o_edge->dst(), cur_partition_nums_);
                g->RemoveEdge(axis_edge);
                o_edge->dst()->ClearAttr("N");
                o_edge->dst()->AddAttr("N", cur_partition_nums_);
              }
              if (o_edge->dst_input() != i) {
                g->AddEdge(o_node, 0, o_edge->dst(), i);
                g->RemoveEdge(o_edge);
              }
            }
          }
        }
      }
    } else {
      for (auto* o_node : ev_node->out_nodes()) {
        if (o_node->type_string() == kIdentityOp) {
          for (auto* oo_node : o_node->out_nodes()) {
            if (oo_node->type_string() == "ReAssign") {
              nodes_to_delete.emplace(oo_node);
            }
          }
        }
      }
    }
  }

  return Status::OK();
}

Status ElasticTrainingPass::ScalingDownRedistributionGraph(
    VarType& var_type, Graph* g, std::vector<Node*>& ev_node_vec, 
    std::unordered_set<Node*>& nodes_to_delete,
    int ev_partition_num) {
  Status s;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      s = ScalingDownEVRedistributionGraph(g, ev_node_vec, nodes_to_delete,
                                           ev_partition_num);
      break;
    case VarType::RESOURCE_VAR:
      s = ScalingDownResVarRedistributionGraph(g, ev_node_vec, nodes_to_delete,
                                              ev_partition_num);
      break;
    default:
      s = ScalingDownVarRedistributionGraph(g, ev_node_vec, nodes_to_delete,
                                            ev_partition_num);
      break;
  }
  return s;
}

Status ElasticTrainingPass::ScalingDownEVRedistributionGraph(
    Graph* g, std::vector<Node*>& ev_node_vec, std::unordered_set<Node*>& nodes_to_delete,
    int ev_partition_num) {
  Status s;
  DataType key_type, value_type;
  std::vector<Node*> filtered_node_vec;
  filtered_node_vec.reserve(cur_partition_nums_);

  for (int i = 0; i < ev_partition_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == kEvExportOp) {
        TF_RETURN_IF_ERROR(GetNodeAttr(o_node->attrs(), "Tkeys", &key_type));
        TF_RETURN_IF_ERROR(
            GetNodeAttr(o_node->attrs(), "dtype", &value_type));
        if (i < cur_partition_nums_) {
          filtered_node_vec.push_back(o_node);
        } else {
          nodes_to_delete.insert(o_node);
        }
      }
    }
  }

  for (int i = 0; i < ev_partition_num; ++i) {
    auto* ev_node = ev_node_vec[i];
    for (auto* o_node : ev_node->out_nodes()) {
      if (o_node->type_string() == kEvImportOp) {
        if (i < cur_partition_nums_) {
          string import_op_name = o_node->name();
          nodes_to_delete.insert(o_node);
          std::vector<NodeDefBuilder::NodeOut> import_keys;
          std::vector<NodeDefBuilder::NodeOut> import_values;
          std::vector<NodeDefBuilder::NodeOut> import_versions;
          std::vector<NodeDefBuilder::NodeOut> import_freqs;
          for (int j = 0; j < filtered_node_vec.size(); ++j) {
            if (j != i) {
              import_keys.emplace_back(filtered_node_vec[j]->name(), 0,
                                       filtered_node_vec[j]->output_type(0));
              import_values.emplace_back(filtered_node_vec[j]->name(), 1,
                                         filtered_node_vec[j]->output_type(1));
              import_versions.emplace_back(
                  filtered_node_vec[j]->name(), 2,
                  filtered_node_vec[j]->output_type(2));
              import_freqs.emplace_back(filtered_node_vec[j]->name(), 3,
                                        filtered_node_vec[j]->output_type(3));
            }
          }
          NodeDef import_storage_node_def;
          TF_RETURN_IF_ERROR(
              NodeDefBuilder(import_op_name + "/Import" , kEvImportOp)
                  .Input(ev_node->name(), 0, ev_node->output_type(0))
                  .Input(import_keys)
                  .Input(import_values)
                  .Input(import_versions)
                  .Input(import_freqs)
                  .Attr("partition_id", i)
                  .Attr("partition_nums", cur_partition_nums_)
                  .Attr("Tkeys", key_type)
                  .Attr("dtype", value_type)
                  .Finalize(&import_storage_node_def));
          Node* import_node = g->AddNode(import_storage_node_def, &s);
          TF_RETURN_IF_ERROR(s);
          import_node->set_assigned_device_name(ev_node->assigned_device_name());
        } else {
          nodes_to_delete.insert(o_node);
        }
      }
    }
  }

  return s;
}

Status ElasticTrainingPass::RewriteElasticPartitionGraph(
    VarType var_type, Graph* g, std::vector<Node*>& ev_node_vec, Node** elastic_node,
    Node** p_dynamic_stitch_node, std::unordered_set<Node*>& nodes_to_delete) {
  LOG(INFO) << "Calling RewriteElasticPartitionGraph --------- ";
  Status s;
  Node* dynamic_partition_node = nullptr;
  Node* dynamic_stitch_node = nullptr;
  std::vector<Node*> identity_node_vec;
  std::vector<Node*> gather_node_vec;

  string gather_name;
  switch (var_type) {
    case VarType::EMBEDDING_VAR:
      gather_name = "KvResourceGather";
      for (int i = 0; i < cur_partition_nums_; ++i) {
        auto* ev_node = ev_node_vec[i];
        for (auto* o_node : ev_node->out_nodes()) {
          if (o_node->type_string() == gather_name) {
            gather_node_vec.push_back(o_node);
            const Edge* input_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(1, &input_edge));
            if (input_edge->src()->type_string() == kDynamicPartition) {
              dynamic_partition_node = input_edge->src();
            }

            for (auto* oo_node : o_node->out_nodes()) {
              if (oo_node->type_string() == kIdentityOp) {
                identity_node_vec.push_back(oo_node);
                for (auto* ooo_node : oo_node->out_nodes()) {
                  if (ooo_node->type_string() == kParaDynamicStitch) {
                    dynamic_stitch_node = ooo_node;
                  }
                }
              }
            }
          }
        }
      }
      break;
    case VarType::REF_VAR:
      gather_name = "GatherV2";
      for (int i = 0; i < cur_partition_nums_; ++i) {
        auto* ev_node = ev_node_vec[i];
        for (auto* o_node : ev_node->out_nodes()) {
          if (o_node->type_string() == kIdentityOp) {
            for (auto* oo_node : o_node->out_nodes()) {
              if (oo_node->type_string() == gather_name) {
                gather_node_vec.push_back(oo_node);
                identity_node_vec.push_back(oo_node); // trick
                if (i == 0) {
                  const Edge* input_edge = nullptr;
                  TF_RETURN_IF_ERROR(oo_node->input_edge(1, &input_edge));
                  if (input_edge->src()->type_string() == kDynamicPartition) {
                    dynamic_partition_node = input_edge->src();
                  }

                  for (auto* ooo_node : oo_node->out_nodes()) {
                    if (ooo_node->type_string() == kParaDynamicStitch) {
                      dynamic_stitch_node = ooo_node;
                    }
                  }
                }
              }
            }
          }
        }
      }
      break;
    case VarType::RESOURCE_VAR:
      gather_name = "ResourceGather";
      for (int i = 0; i < cur_partition_nums_; ++i) {
        auto* ev_node = ev_node_vec[i];
        for (auto* o_node : ev_node->out_nodes()) {
          if (o_node->type_string() == gather_name) {
            gather_node_vec.push_back(o_node);
            const Edge* input_edge = nullptr;
            TF_RETURN_IF_ERROR(o_node->input_edge(1, &input_edge));
            if (input_edge->src()->type_string() == kDynamicPartition) {
              dynamic_partition_node = input_edge->src();
            }

            for (auto* oo_node : o_node->out_nodes()) {
              if (oo_node->type_string() == kIdentityOp) {
                identity_node_vec.push_back(oo_node);
                for (auto* ooo_node : oo_node->out_nodes()) {
                  if (ooo_node->type_string() == kParaDynamicStitch) {
                    dynamic_stitch_node = ooo_node;
                  }
                }
              }
            }
          }
        }
      }
      break;
    default: // DENSE_LAYER_VAR
      return Status::OK();
  }

  if ((dynamic_stitch_node == nullptr) || (dynamic_partition_node == nullptr)) {
    return errors::Internal(
        "dynamic_stitch_node or dynamic_partition_node is nullptr");
  }

  std::string node_name = dynamic_partition_node->name();
  DataType key_type;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(dynamic_partition_node->attrs(), "T", &key_type));
  int num_partitions;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(dynamic_partition_node->attrs(),
                  "num_partitions", &num_partitions));
  const Node* a_copy;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_node(0, &a_copy));
  const Node* b_copy;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_node(1, &b_copy));
  auto idx = node_name.find(kDynamicPartition);
  std::string pre_node_name = node_name.substr(0, idx-1);
  NodeDef elastic_node_def;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder(pre_node_name + "/ElasticPartition", "ElasticPartition")
          .Input(a_copy->name(), 0, a_copy->output_type(0))
          .Input(b_copy->name(), 0, b_copy->output_type(0))
          .Attr("num_partitions", cur_partition_nums_)
          .Attr("TKey", key_type)
          .Finalize(&elastic_node_def));
  *elastic_node = g->AddNode(elastic_node_def, &s);
  TF_RETURN_IF_ERROR(s);
  (*elastic_node)->set_assigned_device_name(dynamic_partition_node->assigned_device_name());

  const Edge* input_edge = nullptr;
  TF_RETURN_IF_ERROR(dynamic_partition_node->input_edge(1, &input_edge));
  for (auto* o_node : input_edge->src()->out_nodes()) {
    if (o_node->type_string() == kDynamicPartition) {
      const Edge* data_input_edge = nullptr;
      TF_RETURN_IF_ERROR(o_node->input_edge(0, &data_input_edge));
      if (data_input_edge->src()->type_string() != "Range") {  // ID
        // Input
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(),
                   *elastic_node, 0);
        nodes_to_delete.insert(o_node);

      } else {  // Indices
        // Input
        g->AddEdge(data_input_edge->src(), data_input_edge->src_output(),
                   *elastic_node, 1);
        nodes_to_delete.insert(o_node);
      }
    }
  }


  *p_dynamic_stitch_node = CopyNode(g, dynamic_stitch_node,
                                    dynamic_stitch_node->assigned_device_name(), 0);
  int part_num;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(dynamic_stitch_node->attrs(), "N", &part_num));
  (*p_dynamic_stitch_node)->ClearAttr("N");
  (*p_dynamic_stitch_node)->AddAttr("N", cur_partition_nums_);
  nodes_to_delete.insert(dynamic_stitch_node);
  for (auto* o_edge : dynamic_stitch_node->out_edges()) {
    // TF_RETURN_IF_ERROR(g->UpdateEdge(*p_dynamic_stitch_node, o_edge->src_output(), o_edge->dst(),
    //               o_edge->dst_input()));
    g->AddEdge(*p_dynamic_stitch_node, o_edge->src_output(), o_edge->dst(),
                  o_edge->dst_input());
  }

  for (int i = 0; i < identity_node_vec.size(); ++i) {
    g->AddEdge(*elastic_node, cur_partition_nums_ + i, *p_dynamic_stitch_node,
               i);
    g->AddEdge(identity_node_vec[i], 0, *p_dynamic_stitch_node,
               cur_partition_nums_ + i);
  }

  for (int i = 0; i < gather_node_vec.size(); ++i) {
    if (i < part_num) {
      TF_RETURN_IF_ERROR(
          g->UpdateEdge(*elastic_node, i, gather_node_vec[i], 1));
    } else {
      g->AddEdge(*elastic_node, i, gather_node_vec[i], 1);
    }
  }
  nodes_to_delete.insert(input_edge->src());
  return s;
}

Status ElasticTrainingPass::UpdatePartitionNums(int& partition_nums) {
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
  partition_nums = ps_array.size();
  LOG(INFO) << " partition_nums is " << partition_nums;
  return Status::OK();
}

Status ElasticTrainingPass::InitHookMetaNode(Graph* g, ElasticHookMetaNode& meta_node) {
  Node* dataset_init = nullptr;
  for (auto* node : g->nodes()) {
    if (node->name() == "elastic_subgraph_import") {
      meta_node.m_import_op_main = node;
    } else if (node->name() == "elastic_subgraph_init") {
      meta_node.m_init_op_main = node;
    } else if (node->name() == "make_initializer") {
      dataset_init = node;
    }
  }

  if ((dataset_init != nullptr) && 
      (meta_node.m_init_op_main != nullptr)) {
    g->AddControlEdge(dataset_init, meta_node.m_init_op_main);
  }

  Status s;
  for (int i = 0; i < cur_partition_nums_; ++i) {
    string new_device_name = "/job:ps/replica:0/task:" +
                             std::to_string(i) + "/device:CPU:0";
    NodeDef initop_def;
    TF_RETURN_IF_ERROR(
        NodeDefBuilder("new_sub_graph/InitOp_" + std::to_string(i),
                        "NoOp")
            .Finalize(&initop_def));
    Node* init_node = g->AddNode(initop_def, &s);
    init_node->set_assigned_device_name(new_device_name);
    TF_RETURN_IF_ERROR(s);
    meta_node.m_init_op_vec[i] = init_node;
    g->AddControlEdge(meta_node.m_init_op_vec[i], meta_node.m_init_op_main);
  }

  NodeDef initop_def;
  TF_RETURN_IF_ERROR(
      NodeDefBuilder("new_sub_graph/tmp_value/InitOp", "NoOp")
          .Finalize(&initop_def));
  meta_node.m_tmp_value_init_op = g->AddNode(initop_def, &s);
  TF_RETURN_IF_ERROR(s);
  g->AddControlEdge(meta_node.m_tmp_value_init_op, meta_node.m_init_op_main);
  return s;
}

Status ElasticTrainingPass::InitVarMeta(
    Graph* g, std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
    std::unordered_map<std::string, std::vector<std::string>>&
        primary_node_to_opt_map,
    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
    std::unordered_map<std::string, Node*>& unpartitioned_node_map) {
  for (auto* node : g->op_nodes()) {
    string node_name = node->name();
    int device_idx;
    int partiton_size;
    VarType var_type;
    if (node->IsKvVarHandle()) {
      var_type = VarType::EMBEDDING_VAR;
    } else if (node->IsVariable()) {
      if (IsRefType(node->output_type(0))) {
        var_type = VarType::DENSE_REF_VAR;
        TensorShape tensor_shape;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &tensor_shape));
        partiton_size = tensor_shape.dim_size(0);
        TF_RETURN_IF_ERROR(FindNode(node, kIdentityOp,
            [this, &var_type](Node* target_node) {
              for (auto* oo_node : target_node->out_nodes()) {
                if (oo_node->type_string() == "GatherV2") {
                  var_type = VarType::REF_VAR;
                }
              }
              return Status::OK();
            }));
        TF_RETURN_IF_ERROR(FindNode(node, "ReAssign",
          [this, &device_idx](Node* target_node) {
            TF_RETURN_IF_ERROR(
              GetNodeAttr(target_node->attrs(), "partition_id", &device_idx));
            return Status::OK();
          }));
      }
    } else if (node->type_string() == "VarHandleOp") {
      var_type = VarType::DENSE_RESOUCE_VAR; 
      TF_RETURN_IF_ERROR(FindNode(node, "ResourceGather",
          [this, &var_type](Node* target_node) {
            var_type = VarType::RESOURCE_VAR;
            return Status::OK();
          }));
      TF_RETURN_IF_ERROR(FindNode(node, "ReAssignResource",
          [this, &device_idx](Node* target_node) {
            TF_RETURN_IF_ERROR(
              GetNodeAttr(target_node->attrs(), "partition_id", &device_idx));
            return Status::OK();
          }));
    } else {
      continue;
    }

    if (node_name.find(kPart) != string::npos) {
      auto part_idx = node_name.find(kPart);
      std::string pre_str = node_name.substr(0, part_idx - 1);
      std::string post_str = node_name.substr(part_idx + strlen(kPart));
      auto post_idx = post_str.find("/");
      if (post_idx == string::npos) {
        if (var_type == VarType::EMBEDDING_VAR) device_idx = std::stoi(post_str);
        if (primary_node_metas_map.find(pre_str) ==
            primary_node_metas_map.end()) {
          PartitionVarMeta var_meta;
          var_meta.m_var_type = var_type;
          var_meta.m_full_shape = partiton_size;
          var_meta.m_partition_num = 1;
          primary_node_metas_map.emplace(pre_str, std::move(var_meta));
          std::vector<Node*> ev_vec(cur_partition_nums_ + 20 /* hack*/,
                                    nullptr);
          ev_vec[device_idx] = node;
          node_to_origin_map.emplace(pre_str, std::move(ev_vec));
        } else {
          primary_node_metas_map[pre_str].m_full_shape += partiton_size;
          primary_node_metas_map[pre_str].m_partition_num++;
          node_to_origin_map[pre_str][device_idx] = node;
        }
        // exactly once
        if (device_idx == 0) {
          if (primary_node_to_opt_map.find(pre_str) ==
              primary_node_to_opt_map.end()) {
            primary_node_to_opt_map.emplace(pre_str, std::vector<string>{});
          }
        }
      } else {
        if (var_type == VarType::EMBEDDING_VAR) device_idx = std::stoi(post_str.substr(0, post_idx));
        string opt_name = pre_str + post_str.substr(post_idx);
        if (primary_node_metas_map.find(opt_name) ==
            primary_node_metas_map.end()) {
          PartitionVarMeta var_meta;
          var_meta.m_var_type = var_type;
          var_meta.m_full_shape = partiton_size;
          var_meta.m_partition_num = 1;
          primary_node_metas_map.emplace(opt_name, std::move(var_meta));
          std::vector<Node*> ev_vec(cur_partition_nums_ + 20 /* hack*/,
                                    nullptr);
          ev_vec[device_idx] = node;
          node_to_origin_map.emplace(opt_name, std::move(ev_vec));
        } else {
          primary_node_metas_map[opt_name].m_full_shape += partiton_size;
          primary_node_metas_map[opt_name].m_partition_num++;
          node_to_origin_map[opt_name][device_idx] = node;
        }
        // exactly once
        if (device_idx == 0) {
          auto sep_idx = opt_name.rfind("/");
          string primary_ev_name = opt_name.substr(0, sep_idx);
          if (primary_node_to_opt_map.find(pre_str) ==
              primary_node_to_opt_map.end()) {
            primary_node_to_opt_map.emplace(pre_str,
                                            std::vector<string>{opt_name});
          } else {
            primary_node_to_opt_map[pre_str].emplace_back(opt_name);
          }
        }
      }
    } else {
      if (node_name.find(kElasticImportScope) == string::npos) {
        unpartitioned_node_map.emplace(node_name, node);
      }
    }
  }
  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0, ElasticTrainingPass);

} // namespace tensorflow
