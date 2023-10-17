#ifndef DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_H_
#define DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"

namespace tensorflow {

enum VarType {
  EMBEDDING_VAR = 1,
  RESOURCE_VAR = 2,
  REF_VAR = 3,
  DENSE_RESOUCE_VAR = 4,
  DENSE_REF_VAR = 5,
};

struct PartitionVarMeta {
  VarType m_var_type;
  int m_full_shape;
  int m_partition_num;
};

struct ElasticHookMetaNode {
  Node* m_import_op_main;
  Node* m_init_op_main;
  Node* m_tmp_value_init_op;
  std::vector<Node*> m_init_op_vec;

  ElasticHookMetaNode(int num_partition): 
      m_import_op_main(nullptr), m_init_op_main(nullptr),
      m_tmp_value_init_op(nullptr), m_init_op_vec(num_partition, nullptr) {};
};

class ElasticTrainingPass : public GraphOptimizationPass {
  public:
    Status Run(const GraphOptimizationPassOptions& options) override;

    Status RewriteSubGraph(Graph* g, bool is_test = false);

    Status InitHookMetaNode(Graph* g, ElasticHookMetaNode& meta_node);

    Status InitVarMeta(Graph* g,
                       std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
                       std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                       std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                       std::unordered_map<std::string, Node*>& unpartitioned_node_map);

    Status RewriteTrainingSubGraph(Graph* g,
                                   std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
                                   std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                   std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                   ElasticHookMetaNode& meta_node,
                                   bool is_test);
    
    Status ScalingDownRedistributionGraph(VarType& var_type, Graph* g,
                                          std::vector<Node*>& new_ev_node_vec,
                                          std::unordered_set<Node*>& nodes_to_delete,
                                          int ev_partition_num);

    Status RewriteSavingSubGraph(Graph* g,
                                std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
                                std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                std::unordered_map<std::string, Node*>& unpartitioned_node_map,
                                ElasticHookMetaNode& meta_node);
    
    Status ScalingUpForWardGraph(const VarType& var_type, Graph* g, 
                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                  std::unordered_set<Node*>& nodes_to_delete,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  ElasticHookMetaNode& meta_node, 
                                  int part_var_full_shape, int ev_partition_num);

    Status ScalingUpRedistributionGraph(VarType var_type, Graph* g,
                                        std::vector<Node*>& new_ev_node_vec, Node* import_op_main,
                                        int ev_partition_num, std::vector<Node*>& primary_ev_filters);
    
    Status RewriteElasticPartitionGraph(VarType var_type,
                                        Graph* g, 
                                        std::vector<Node*>& ev_node_vec,
                                        Node** elastic_node,
                                        Node** p_dynamic_stitch_node,
                                        std::unordered_set<Node*>& nodes_to_delete);

    Status ScalingUpBackWardGraph(VarType var_type, Graph* g, 
                                  std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  Node* elastic_node, Node* p_dynamic_stitch_node,
                                  std::vector<Node*>& no_op_vec, int part_var_full_shape,
                                  int ev_partition_num);
    
    Status ScalingDownForWardGraph(const VarType& var_type, Graph* g, 
                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                  std::unordered_set<Node*>& nodes_to_delete,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  int part_var_full_shape,
                                  int ev_partition_num);

    Status ScalingDownBackWardGraph(Graph* g, VarType var_type,
                                    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                    std::unordered_set<Node*>& nodes_to_delete,
                                    const std::string& primary_ev_name,
                                    const std::vector<std::string>& opt_ev_names,
                                    Node* elastic_node, Node* p_dynamic_stitch_node,
                                    int part_var_full_shape, int ev_partition_num);

  private:

    Status UpdatePartitionNums(int& partition_nums);

    Status ScalingUpVarForWardGraph(const VarType& var_type,
                                    Graph* g, 
                                    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                    std::unordered_set<Node*>& nodes_to_delete,
                                    const std::string& primary_ev_name,
                                    const std::vector<std::string>& opt_ev_names,
                                    ElasticHookMetaNode& meta_node, int ev_partition_num);

    Status ScalingUpEVForWardGraph(const VarType& var_type,
                                   Graph* g, 
                                   std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                   std::unordered_set<Node*>& nodes_to_delete,
                                   const std::string& primary_ev_name,
                                   const std::vector<std::string>& opt_ev_names,
                                   ElasticHookMetaNode& meta_node, int ev_partition_num);
    
    Status ScalingUpResVarForWardGraph(const VarType& var_type,
                                   Graph* g, 
                                   std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                   std::unordered_set<Node*>& nodes_to_delete,
                                   const std::string& primary_ev_name,
                                   const std::vector<std::string>& opt_ev_names,
                                   ElasticHookMetaNode& meta_node, 
                                   int part_var_full_shape, int ev_partition_num);

    Status ScalingUpEVRedistributionGraph(VarType var_type, Graph* g,
                                        std::vector<Node*>& new_ev_node_vec, Node* import_op_main,
                                        int ev_partition_num, std::vector<Node*>& primary_ev_filters);
    
    Status ScalingUpResVarRedistributionGraph(VarType var_type, Graph* g,
                                        std::vector<Node*>& new_ev_node_vec, Node* import_op_main,
                                        int ev_partition_num, std::vector<Node*>& primary_ev_filters);

    Status ScalingUpVarRedistributionGraph(VarType var_type, Graph* g,
                                        std::vector<Node*>& new_ev_node_vec, Node* import_op_main,
                                        int ev_partition_num, std::vector<Node*>& primary_ev_filters);

    Status ScalingUpEmbeddingVariableBackWardGraph(VarType var_type, Graph* g, 
                                  std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  Node* elastic_node, Node* p_dynamic_stitch_node,
                                  std::vector<Node*>& no_op_vec,
                                  int part_var_full_shape, int ev_partition_num);
    
    Status ScalingDownEVForWardGraph(const VarType& var_type, Graph* g, 
                                      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                      std::unordered_set<Node*>& nodes_to_delete,
                                      const std::string& primary_variable_name,
                                      const std::vector<std::string>& opt_ev_names,
                                      int ev_partition_num);

    Status ScalingDownResVarForWardGraph(const VarType& var_type, Graph* g, 
                                      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                      std::unordered_set<Node*>& nodes_to_delete,
                                      const std::string& primary_variable_name,
                                      const std::vector<std::string>& opt_ev_names,
                                      int part_var_full_shape,
                                      int ev_partition_num);

    Status ScalingDownVarForWardGraph(const VarType& var_type, Graph* g, 
                                      std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                      std::unordered_set<Node*>& nodes_to_delete,
                                      const std::string& primary_variable_name,
                                      const std::vector<std::string>& opt_ev_names,
                                      int ev_partition_num);

    Status ScalingUpDenseBackWardGraph(VarType var_type, Graph* g, 
                                  std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  Node* elastic_node, 
                                  std::vector<Node*>& no_op_vec, int part_var_full_shape,
                                  int ev_partition_num);

    Status ScalingDownEVRedistributionGraph(Graph* g,
                                          std::vector<Node*>& new_ev_node_vec,
                                          std::unordered_set<Node*>& nodes_to_delete,
                                          int ev_partition_num);

    Status ScalingDownResVarRedistributionGraph(Graph* g,
                                          std::vector<Node*>& new_ev_node_vec,
                                          std::unordered_set<Node*>& nodes_to_delete,
                                          int ev_partition_num);

    Status ScalingDownVarRedistributionGraph(Graph* g,
                                          std::vector<Node*>& new_ev_node_vec,
                                          std::unordered_set<Node*>& nodes_to_delete,
                                          int ev_partition_num);

    Status ScalingDownEmbeddingVariableBackWardGraph(VarType var_type, Graph* g, 
                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                  std::unordered_set<Node*>& nodes_to_delete,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  Node* elastic_node, Node* p_dynamic_stitch_node,
                                  int part_var_full_shape, int ev_partition_num);

    Status ScalingDownDenseBackWardGraph(Graph* g,
                                  std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                  std::unordered_set<Node*>& nodes_to_delete,
                                  const std::string& primary_ev_name,
                                  Node* elastic_node,
                                  int part_var_full_shape, int ev_partition_num);

    Status RewritePrevSubGraph(Graph* g,
                                int i, std::vector<Node*>& save_node_vec, 
                                std::unordered_set<Node*>& nodes_to_delete,
                                std::unordered_map<string, std::vector<int64>>& variable_shape,
                                std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
                                std::unordered_set<Node*>& eval_nodes_to_add,
                                std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map);

    Status DeleteOldSaverGraph(std::vector<Node*>& save_node_vec, int i,
                              std::unordered_map<Node*, std::pair<string, string>>& nodes_to_add,
                              std::unordered_set<Node*>& nodes_to_delete,
                              std::unordered_map<string, std::vector<int64>>& variable_shape,
                              std::unordered_map<std::string, Node*>& unpartitioned_node_map);

    Status AddNewSaverGraph(Graph* g, bool& has_ev, Node** new_sharded_filename,
                            std::vector<string>& tensor_names_vec,
                            std::vector<DataType>& n_dtypes,
                            std::unordered_map<std::string, PartitionVarMeta>& primary_node_metas_map,
                            std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                            std::unordered_map<string, std::vector<int64>>& variable_shape,
                            std::vector<Node*>& restore_tensor_vec,
                            std::string& assigned_device_name,
                            std::vector<Node*>& save_node_vec,
                            int i);
    
    Status AddNewRestoreGraph(Graph* g,
                              bool has_ev, Node* new_sharded_filename,
                              std::vector<string>& tensor_names_vec,  
                              std::vector<DataType>& n_dtypes,
                              std::unordered_map<string, std::vector<int64>> variable_shape,
                              const std::string& assigned_device_name,
                              std::vector<Node*>& restore_tensor_vec,
                              std::vector<Node*>& restore_node_vec,
                              int i);
  private:
    static int cur_partition_nums_;
    bool scaling_up_{false};
};

}
#endif // DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_H_
