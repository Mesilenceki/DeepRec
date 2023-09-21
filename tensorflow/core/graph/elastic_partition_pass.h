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
  DENSE_LAYER_VAR = 4,
};

struct PartitionVarMeta {
  VarType m_var_type;
  int m_full_shape;
  int m_partition_num;
};

class ElasticTrainingPass : public GraphOptimizationPass {
  public:
    Status Run(const GraphOptimizationPassOptions& options) override;

    Status RewriteSubGraph(Graph* g, bool is_test = false);

    Status InitVarMeta(Graph* g,
                       std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
                       std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                       std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map);
    
    Status RewriteTrainingSubGraph(Graph* g,
                                   std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
                                   std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                   std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                   bool is_test);
    
    Status ScalingDownRedistributionGraph(VarType& var_type, Graph* g,
                                          std::vector<Node*>& new_ev_node_vec,
                                          std::unordered_set<Node*>& nodes_to_delete,
                                          int ev_partition_num);

    Status UpdatePartitionNums(int& partition_nums);
    Status RewriteSavingSubGraph(Graph* g,
                                std::unordered_map<std::string, PartitionVarMeta>& primary_ev_metas_map,
                                std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map);
    
    Status ScalingUpForWardGraph(const VarType& var_type, Graph* g, 
                                  std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                  std::unordered_set<Node*>& nodes_to_delete,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  std::vector<Node*>& init_op_vec, int ev_partition_num);

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
                                  int ev_partition_num);

    Status ScalingDownBackWardGraph(Graph* g, VarType var_type,
                                    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                    std::unordered_set<Node*>& nodes_to_delete,
                                    const std::string& primary_ev_name,
                                    const std::vector<std::string>& opt_ev_names,
                                    Node* elastic_node, Node* p_dynamic_stitch_node,
                                    int part_var_full_shape, int ev_partition_num);

  private:

    Status ScalingUpVarForWardGraph(const VarType& var_type,
                                    Graph* g, 
                                    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                    std::unordered_set<Node*>& nodes_to_delete,
                                    const std::string& primary_ev_name,
                                    const std::vector<std::string>& opt_ev_names,
                                    std::vector<Node*>& init_op_vec, int ev_partition_num);

    Status ScalingUpEVForWardGraph(const VarType& var_type,
                                   Graph* g, 
                                   std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                   std::unordered_set<Node*>& nodes_to_delete,
                                   const std::string& primary_ev_name,
                                   const std::vector<std::string>& opt_ev_names,
                                   std::vector<Node*>& init_op_vec, int ev_partition_num);
    
    Status ScalingUpResVarForWardGraph(const VarType& var_type,
                                   Graph* g, 
                                   std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                   std::unordered_set<Node*>& nodes_to_delete,
                                   const std::string& primary_ev_name,
                                   const std::vector<std::string>& opt_ev_names,
                                   std::vector<Node*>& init_op_vec, int ev_partition_num);

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

  private:
    static int cur_partition_nums_;
    bool scaling_up_{false};
};

}
#endif // DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_H_
