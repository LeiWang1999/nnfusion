// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "graph_pass_base.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class NCHW2NHWCPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
            private:
                std::shared_ptr<nnfusion::graph::GNode>
                    dfs_convert_nodes(std::shared_ptr<nnfusion::graph::Graph>& graph,
                                      std::shared_ptr<nnfusion::graph::GNode>& root_node);
                std::vector<std::shared_ptr<nnfusion::graph::GNode>> nodes_4d;
                std::map<std::string, std::shared_ptr<nnfusion::graph::GNode>>
                    nchw2nhwc_nodes;
                std::map<std::string, std::vector<std::string>> nodes_4d_name_inputs;
            };
        } // namespace graph
    }     // namespace pass
} // namespace nnfusion
