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

    Status RewriteTrainingGraph(Graph* g, bool is_test = false);
    Status RewriteElasticPartitionGraph(Graph* g, std::vector<Node*>& ev_node_vec);
    Status InitEVMeta(Graph* g,
                      std::unordered_map<std::string, Node* >& ev_nodes_map,
                      std::unordered_map<std::string, std::pair<bool, int>>& ev_metas_map,
                      std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map);
    
    Status InitNewPartitionSubGraph(Graph* g,
                                    std::unordered_map<std::string, std::pair<bool, int>>& ev_metas_map,
                                    std::unordered_map<std::string, Node*>& ev_nodes_map,
                                    std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map,
                                    bool is_test);
    Status ScalingUpRedistributionGraph(Graph* g,
                                      std::vector<Node*>& new_ev_node_vec, int ev_partition_num);
    
    Status ScalingDownRedistributionGraph(Graph* g,
                                      std::vector<Node*>& new_ev_node_vec, int ev_partition_num);

    Status UpdatePartitionNums();
    Status InitNewSaveSubGraph(Graph* g,
                                std::unordered_map<std::string, std::pair<bool, int>>& ev_metas_map,
                                std::unordered_map<std::string, Node*>& ev_nodes_map,
                                std::unordered_map<std::string, std::vector<Node*>>& ev_to_origin_map);
    
    Status ScalingUpBackWardGraph(Graph* g,
                                  std::vector<Node*>& ev_node_vec,
                                  int ev_partition_num);
                                  
    Status ScalingDownBackWardGraph(Graph* g,
                                  std::vector<Node*>& ev_node_vec,
                                  int ev_partition_num);

              
  private:
    int partition_nums_;
};

}
#endif // DYNMAMIC_EMBEDDING_SERVER_CC_GRAPH_ELASTIC_PARTITION_PASS_H_