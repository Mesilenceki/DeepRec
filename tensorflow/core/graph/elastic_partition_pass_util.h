#ifndef DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_UTIL_H_
#define DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_UTIL_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/elastic_partition_pass.h"

namespace tensorflow {

Status GetPartVariableShape(std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
                        Node* save_node, std::unordered_map<string, std::vector<int64>>& variable_shape,
                        int cur_partition_nums) {
  Node* tensor_name_node;
  TF_RETURN_IF_ERROR(save_node->input_node(1, &tensor_name_node));
  Tensor tensor_name_t;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(tensor_name_node->attrs(), "value", &tensor_name_t));
  Node* shape_and_slice_node;
  TF_RETURN_IF_ERROR(save_node->input_node(2, &shape_and_slice_node));
  Tensor shape_and_slice_t;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(shape_and_slice_node->attrs(), "value", &shape_and_slice_t));
  
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
          if (it->second.m_partition_num == 1) {
            shape_vec[j + items.size()] = dim;
          } else {
            shape_vec[j + items.size()] = dim / cur_partition_nums;
          }
        } else {
          shape_vec[j] = dim;
          shape_vec[j + items.size()] = dim;
        }
      }
      variable_shape.emplace(tensor_n, std::move(shape_vec));
      LOG(INFO) << "variable_shape name: " << tensor_n;
    }
  }
  return Status::OK();
}

Status ScalingSaverSaverNodeUtil(Graph* g, Node* ori_save_node, int i,
                                 std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
                                 std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                 string& assigned_device_name, bool& has_ev,
                                 std::vector<Node*>& kv_lookup_resource_node_vec,
                                 std::vector<string>& ev_names_vec,
                                 std::vector<DataType>& key_data_types,
                                 std::vector<string>& tensor_names_vec,
                                 std::vector<Node*>& restore_tensor_vec,
                                 std::vector<Node*>& tensor_vec,
                                 std::vector<NodeDefBuilder::NodeOut>& tensors_input,
                                 std::vector<DataType> n_dtypes) {
    Status s;
    for (auto& it : primary_node_metas_map) {
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
                } else if (o_node->type_string() == "ReadVariableOp") {
                    for (auto* oo_node : o_node->out_nodes()) {
                        if (oo_node->type_string() == "Identity") { 
                            for (auto* ooo_node : oo_node->out_nodes()) {
                                if (ooo_node->type_string() == "Identity") {
                                    tensor_vec.emplace_back(ooo_node);
                                }
                            }
                        }
                    }
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
        LOG(INFO) << "tensor_name: " << tensor->name();
    }
    return s;
}

}

#endif // DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_UTIL_H_