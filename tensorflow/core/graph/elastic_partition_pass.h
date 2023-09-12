#ifndef DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_H_
#define DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"

namespace tensorflow {
class ElasticTrainingPass : public GraphOptimizationPass {
  public:
    Status Run(const GraphOptimizationPassOptions& options) override;

    Status RewriteSubGraph(Graph* g, bool is_test = false);
    Status RewriteElasticPartitionGraph(Graph* g, 
                                        bool is_ev,
                                        std::vector<Node*>& ev_node_vec,
                                        Node** elastic_node,
                                        Node** p_dynamic_stitch_node);
    Status InitVarMeta(Graph* g,
                       std::unordered_map<std::string, bool>& is_ev_map,
                       std::unordered_map<std::string, int>& primary_ev_metas_map,
                       std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                       std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map);
    
    Status RewriteTrainingSubGraph(Graph* g,
                                   std::unordered_map<std::string, bool>& is_ev_map,
                                   std::unordered_map<std::string, int>& primary_ev_metas_map,
                                   std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                   std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                   bool is_test);
    Status ScalingUpRedistributionGraph(Graph* g,
                                      std::vector<Node*>& new_ev_node_vec, Node* import_op_main,
                                      int ev_partition_num, std::vector<Node*>& primary_ev_filters);
    
    Status ScalingUpVarRedistributionGraph(Graph* g,
                                            std::vector<Node*>& var_node_vec,
                                            Node* import_op_main,
                                            int ev_partition_num);
    
    Status ScalingDownRedistributionGraph(Graph* g,
                                          std::vector<Node*>& new_ev_node_vec, int ev_partition_num);
                                          
    Status ScalingDownVarRedistributionGraph(Graph* g,
                                             std::vector<Node*>& new_ev_node_vec, int ev_partition_num);

    Status UpdatePartitionNums(int& partition_nums);
    Status RewriteSavingSubGraph(Graph* g,
                                std::unordered_map<std::string, bool>& is_ev_map,
                                std::unordered_map<std::string, int>& primary_ev_metas_map,
                                std::unordered_map<std::string, std::vector<std::string>>& primary_ev_to_opt_map,
                                std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map);
    
    Status ScalingUpBackWardGraph(Graph* g, bool is_ev,
                                  std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  Node* elastic_node, Node* p_dynamic_stitch_node,
                                  std::vector<Node*>& no_op_vec,
                                  int ev_partition_num);
    
    Status ScalingDownBackWardGraph(Graph* g, bool is_ev,
                                    std::unordered_map<std::string, std::vector<Node*>>& node_to_origin_map,
                                    const std::string& primary_ev_name,
                                    const std::vector<std::string>& opt_ev_names,
                                    int ev_partition_num);
    
    Status ScalingUpVarBackWardGraph(Graph* g,
                                  std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  std::vector<Node*>& no_op_vec,
                                  int ev_partition_num);

    Status ScalingDownVarBackWardGraph(Graph* g,
                                  std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                  const std::string& primary_ev_name,
                                  const std::vector<std::string>& opt_ev_names,
                                  std::vector<Node*>& no_op_vec,
                                  int ev_partition_num);
  private:
    static int ori_partition_nums_;
    bool scaling_up_{false};
};

}
#endif // DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_H_