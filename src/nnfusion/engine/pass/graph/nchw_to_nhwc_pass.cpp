#include "nchw_to_nhwc_pass.hpp"
#include "gflags/gflags.h"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/graph/graph_util.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"
#include "nnfusion/util/util.hpp"

#include "gflags/gflags.h"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(fnchw_2_nhwc, false, "Enable NCHW 2 NHWC Pass");

std::shared_ptr<GNode>
    NCHW2NHWCPass::dfs_convert_nodes(std::shared_ptr<Graph>& graph,
                                     std::shared_ptr<GNode>& root_node)
{
    auto root_node_op_type = root_node->get_op_type();
    NNFUSION_LOG(INFO) << "root_node_op_type: " << root_node_op_type;
    if (root_node_op_type == "Parameter"){
        // this is the input node, so we need to insert nchw2nhwc transpose node
        NNFUSION_LOG(INFO) << "Parameter " << root_node->get_name();
        nnfusion::op::OpConfig::any nchw2nhwc_transpose_config;
        nchw2nhwc_transpose_config["axes_order"] = {0, 2, 3, 1};
        auto nchw2nhwc_transpose_op = std::make_shared<nnfusion::op::GenericOp>(
            graph->get_name() + "_nchw2nhwc_transpose", "Transpose", nchw2nhwc_transpose_config);
        auto nchw2nhwc_transpose_gnode = graph->add_node_and_edge(nchw2nhwc_transpose_op, GNodeVector{root_node});
        return nchw2nhwc_transpose_gnode;
    }else{
        if (root_node_op_type == "Convolution")
        {
            NNFUSION_LOG(INFO) << "Convolution " << root_node->get_name();
            auto conv_op = static_pointer_cast<nnfusion::op::Convolution>(root_node->get_op_ptr());
            auto strides = conv_op->get_window_movement_strides();
            auto dilations = conv_op->get_window_dilation_strides();
            auto padding_below = conv_op->get_padding_below();
            auto padding_above = conv_op->get_padding_above();
            auto conv_data_format = conv_op->get_data_format();
            assert(conv_data_format == "NCHW");
            // create a new conv node
            auto conv_nhwc_op = std::make_shared<nnfusion::op::Convolution>(
                strides, dilations, padding_below, padding_above, "NHWC");
            
            auto input_edges = root_node->get_in_edges();
            std::vector<std::shared_ptr<GNode>> new_nodes;
            std::vector<std::shared_ptr<GNode>> input_nodes_ptr;
            std::vector<std::string> input_nodes_name = nodes_4d_name_inputs[root_node->get_name()];

            for (auto e : input_edges)
            {
                auto src_node = e->get_src();
                input_nodes_ptr.emplace_back(src_node);
            }

            for (size_t i = 0; i < input_nodes_name.size(); ++i)
            {
                NNFUSION_LOG(INFO) << "input_nodes_name: " << input_nodes_name[i];
                auto name = input_nodes_name[i];
                if (nchw2nhwc_nodes.find(name) != nchw2nhwc_nodes.end())
                {
                    auto src_node = nchw2nhwc_nodes[name];
                    new_nodes.emplace_back(src_node);
                }
                else
                {
                    auto _i = i;
                    if (i >= input_nodes_ptr.size())
                        _i = input_nodes_ptr.size() - 1;
                    auto src_node = input_nodes_ptr[_i];

                    if (src_node->get_op_type() == "Constant")
                    {
                        NNFUSION_LOG(INFO)
                            << "Constant, should change do transpose " << src_node->get_name();
                        auto constant_shape = src_node->get_output_shape(0);
                        auto constant_data_type = src_node->get_output_element_type(0);
                        // convert shape to nhwc
                        auto nhwc_shape = nnfusion::Shape({constant_shape[0],
                                                           constant_shape[2],
                                                           constant_shape[3],
                                                           constant_shape[1]});
                        // todo(v-leiwang3): permuate the data.
                        auto constant_data = src_node->get_output_tensor_ptr(0);
                        auto constant_op =
                            dynamic_pointer_cast<nnfusion::op::Constant>(src_node->get_op_ptr());
                        // get data
                        half_float::half* data = (half_float::half*)constant_op->get_data_ptr();
                        // create a temp storage
                        half_float::half* temp_data =
                            (half_float::half*)(new char[constant_op->get_data_size()]);
                        // copy data
                        memcpy(temp_data, data, constant_op->get_data_size());
                        // permuate data
                        for (size_t i = 0; i < constant_shape[0]; ++i)
                        {
                            for (size_t j = 0; j < constant_shape[1]; ++j)
                            {
                                for (size_t k = 0; k < constant_shape[2]; ++k)
                                {
                                    for (size_t l = 0; l < constant_shape[3]; ++l)
                                    {
                                        size_t idx = i * constant_shape[1] * constant_shape[2] *
                                                         constant_shape[3] +
                                                     j * constant_shape[2] * constant_shape[3] +
                                                     k * constant_shape[3] + l;
                                        size_t idx_nhwc = i * constant_shape[2] * constant_shape[3] *
                                                             constant_shape[1] +
                                                         k * constant_shape[3] * constant_shape[1] +
                                                         l * constant_shape[1] + j;
                                        data[idx_nhwc] = temp_data[idx];
                                    }
                                }
                            }
                        }

                        NNFUSION_LOG(INFO) << "constant_shape: " << constant_shape;
                        NNFUSION_LOG(INFO) << "nhwc_shape: " << nhwc_shape;
                        src_node->set_output_type_and_shape(0, constant_data_type, nhwc_shape);
                        NNFUSION_LOG(INFO) << "new node shape " << src_node->get_output_shape(0);
                        new_nodes.emplace_back(src_node);
                    }
                    else
                    {
                        auto new_node = dfs_convert_nodes(graph, src_node);
                        NNFUSION_LOG(INFO) << "new node shape " << new_node->get_output_shape(0);
                        new_nodes.emplace_back(new_node);
                    }
                }
            }
            NNFUSION_LOG(INFO) << "new_nodes size: " << new_nodes.size();
            NNFUSION_LOG(INFO) << "conv_op: " << conv_op->get_name();
            NNFUSION_LOG(INFO) << "conv_nhwc_op: " << conv_nhwc_op->get_name();
            NNFUSION_LOG(INFO) << "conv_nhwc_op: " << conv_nhwc_op->get_data_format();
            NNFUSION_LOG(INFO) << "input_node 0 " << new_nodes[0]->get_name() << " shape: " << new_nodes[0]->get_output_shape(0);
            NNFUSION_LOG(INFO) << "input_node 1 " << new_nodes[1]->get_name() << " shape: " << new_nodes[1]->get_output_shape(0);

            auto conv_nhwc_gnode = graph->add_node_and_edge(conv_nhwc_op, new_nodes);
            nchw2nhwc_nodes[root_node->get_name()] = conv_nhwc_gnode;
            graph->remove_node(root_node);
            NNFUSION_LOG(INFO) << "conv_nhwc_gnode insert: " << conv_nhwc_gnode->get_out_edges().size();
            return conv_nhwc_gnode;
        }
        else if (root_node_op_type == "Broadcast")
        {
            NNFUSION_LOG(INFO) << "Broadcast";
            auto broadcast_op =
                static_pointer_cast<nnfusion::op::Broadcast>(root_node->get_op_ptr());
            auto broadcast_axes = broadcast_op->get_broadcast_axes();
            auto output_shape = root_node->get_output_shape(0);
            auto broadcast_nhwc_axes = nnfusion::AxisSet({0, 1, 2});
            auto output_nhwc_shape = nnfusion::Shape({output_shape[0], output_shape[2], output_shape[3], output_shape[1]});
            auto broadcast_nhwc_op =
                std::make_shared<nnfusion::op::Broadcast>(output_nhwc_shape, broadcast_nhwc_axes);
            auto input_edges = root_node->get_in_edges();
            std::vector<std::shared_ptr<GNode>> new_nodes;

            for (auto& e : input_edges)
            {
                auto src_node = e->get_src();
                if (src_node->get_op_type() == "Constant")
                {
                    NNFUSION_LOG(INFO)
                        << "Constant, should change do nothing " << src_node->get_name();
                    new_nodes.emplace_back(src_node);
                }
                else
                {
                    new_nodes.emplace_back(dfs_convert_nodes(graph, src_node));
                }
            }
            auto broadcast_nhwc_gnode = graph->add_node_and_edge(broadcast_nhwc_op, new_nodes);
            nchw2nhwc_nodes[root_node->get_name()] = broadcast_nhwc_gnode;
            graph->remove_node(root_node);
            return broadcast_nhwc_gnode;
        }
        else if (root_node_op_type == "Add")
        {
            auto add_op = static_pointer_cast<nnfusion::op::Add>(root_node->get_op_ptr());
            NNFUSION_LOG(INFO) << "Adds " << add_op->get_name();
            auto add_nhwc_op = std::make_shared<nnfusion::op::Add>();

            auto input_edges = root_node->get_in_edges();
            std::vector<std::shared_ptr<GNode>> new_nodes;
            std::vector<std::shared_ptr<GNode>> input_nodes_ptr;
            std::vector<std::string> input_nodes_name;

            for (auto e : input_edges){
                auto src_node = e->get_src();
                input_nodes_ptr.emplace_back(src_node);
                input_nodes_name.emplace_back(src_node->get_name());
            }

            for (size_t i = 0; i < input_nodes_ptr.size(); ++i)
            {
                auto name = input_nodes_name[i];
                auto i_node_ptr = input_nodes_ptr[i];
                // if input nodes has been transposed
                NNFUSION_LOG(INFO) << "process node: " << i_node_ptr->get_name()
                                   << " op_type: " << i_node_ptr->get_op_type();
                if (nchw2nhwc_nodes.find(name) != nchw2nhwc_nodes.end())
                {
                    i_node_ptr = nchw2nhwc_nodes[name];
                    new_nodes.emplace_back(i_node_ptr);
                }
                else
                {
                    new_nodes.emplace_back(dfs_convert_nodes(graph, i_node_ptr));
                }
            }

            NNFUSION_LOG(INFO) << "add_op " << add_op->get_name();
            NNFUSION_LOG(INFO) << "new_nodes size: " << new_nodes.size();
            NNFUSION_LOG(INFO) << "add_nhwc_op: " << add_nhwc_op->get_name();
            NNFUSION_LOG(INFO) << "input_node 0 " << new_nodes[0]->get_name()
                               << " shape: " << new_nodes[0]->get_output_shape(0);
            NNFUSION_LOG(INFO) << "input_node 1 " << new_nodes[1]->get_name()
                               << " shape: " << new_nodes[1]->get_output_shape(0);
            auto add_nhwc_gnode = graph->add_node_and_edge(add_nhwc_op, new_nodes);
            nchw2nhwc_nodes[root_node->get_name()] = add_nhwc_gnode;
            graph->remove_node(root_node);
            return add_nhwc_gnode;
        }
        else if (root_node_op_type == "Relu")
        {
            NNFUSION_LOG(INFO) << "Relu";
            auto relu_op = static_pointer_cast<nnfusion::op::Relu>(root_node->get_op_ptr());
            auto relu_nhwc_op = std::make_shared<nnfusion::op::Relu>();
            NNFUSION_LOG(INFO) << "relu_op " << relu_op->get_name();

            auto input_edges = root_node->get_in_edges();
            std::vector<std::shared_ptr<GNode>> new_nodes;
            for (auto& e : input_edges)
            {
                auto src_node = e->get_src();
                new_nodes.emplace_back(dfs_convert_nodes(graph, src_node));
            }
            NNFUSION_LOG(INFO) << "new_nodes size: " << new_nodes.size();
            NNFUSION_LOG(INFO) << "relu_nhwc_op: " << relu_nhwc_op->get_name();
            NNFUSION_LOG(INFO) << "input_node 0 " << new_nodes[0]->get_name()
                               << " shape: " << new_nodes[0]->get_output_shape(0);
            auto relu_nhwc_gnode = graph->add_node_and_edge(relu_nhwc_op, new_nodes);
            nchw2nhwc_nodes[root_node->get_name()] = relu_nhwc_gnode;
            graph->remove_node(root_node);
            return relu_nhwc_gnode;
        }
        else if (root_node_op_type == "MaxPool")
        {
            NNFUSION_LOG(INFO) << "MaxPool";
            auto maxpool_op = static_pointer_cast<nnfusion::op::MaxPool>(root_node->get_op_ptr());
            auto strides = maxpool_op->get_window_movement_strides();
            auto window_shape = maxpool_op->get_window_shape();
            auto padding_below = maxpool_op->get_padding_below();
            auto padding_above = maxpool_op->get_padding_above();
            auto data_format = maxpool_op->get_data_format();
            assert (data_format == "NCHW");
            auto maxpool_nhwc_op = std::make_shared<nnfusion::op::MaxPool>(
                window_shape, strides, padding_below, padding_above, "NHWC");
            auto input_edges = root_node->get_in_edges();
            std::vector<std::shared_ptr<GNode>> new_nodes;
            for (auto& e : input_edges)
            {
                auto src_node = e->get_src();
                new_nodes.emplace_back(dfs_convert_nodes(graph, src_node));
            }
            NNFUSION_LOG(INFO) << "maxpool_op " << maxpool_op->get_name();
            auto maxpool_nhwc_gnode = graph->add_node_and_edge(maxpool_nhwc_op, new_nodes);
            nchw2nhwc_nodes[root_node->get_name()] = maxpool_nhwc_gnode;
            graph->remove_node(root_node);
            return maxpool_nhwc_gnode;
        }
        else
        {
            NNFUSION_LOG(INFO) << "op_type = " << root_node_op_type
                               << " is not supported the op_name is " << root_node->get_name();
            return nullptr;
        }
    }
    return nullptr;
}

bool NCHW2NHWCPass::run_on_graph(std::shared_ptr<Graph>& graph){
    bool using_pass = FLAGS_fnchw_2_nhwc;
    if (!using_pass)
        return true;
    NNFUSION_LOG(INFO) << "Start NCHW 2 NHWC Pass";
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    // mantain a map of nodes which need to transpose, the common distinction is their output shape is 4d
    for (auto& it : nodes)
    {
        auto node_op_type = it->get_op_type();
        std::vector<std::string> op_type_need_transpose = {
            "Convolution", "MaxPool", "AvgPool", "BatchNorm", "Relu", "Add", "Parameter", "Broadcast"};
        if (std::count(op_type_need_transpose.begin(), op_type_need_transpose.end(), node_op_type))
        {
            auto output_shape = it->get_output_shape(0);
            if (output_shape.size() == 4)
            {
                nodes_4d.emplace_back(it);
                auto input_edges = it->get_in_edges();
                for (const auto edge : input_edges)
                {
                    auto src_node = edge->get_src();
                    nodes_4d_name_inputs[it->get_name()].emplace_back(src_node->get_name());
                }
            }
        }
    }

    for (auto& it : nodes_4d_name_inputs){
        NNFUSION_LOG(INFO) << "node_name: " << it.first;
        for (auto& input_name : it.second){
            NNFUSION_LOG(INFO) << "\tinput_name: " << input_name;
        }
    }

    // print the nodes_4d
    for (size_t node_id; node_id < nodes_4d.size(); node_id++) {
        auto node = nodes_4d[node_id];
        auto node_shape = node->get_output_shape(0);
        auto op_type = node->get_op_type();
        auto op_name = node->get_name();
        if (op_name == "/layer2/layer2.0/downsample/downsample.0/Conv_output_0")
        {
            NNFUSION_LOG(INFO) << "node_id: " << node_id << " op_type: " << op_type
                                << " node_shape: " << node_shape << " node_name "
                                << node->get_name();
            NNFUSION_LOG(INFO) << "input0 " << node->get_in_edge(0)->get_src()->get_name()
                                << " shape: "
                                << node->get_in_edge(0)->get_src()->get_output_shape(0);
            NNFUSION_LOG(INFO) << "input1 " << node->get_in_edge(1)->get_src()->get_name()
                                << " shape: " << node->get_in_edge(1)->get_src()->get_output_shape(0);

        }
    }

    auto last_node_in_nodes_4d = nodes_4d.back();
    auto it = last_node_in_nodes_4d->get_out_edges().front()->get_dst();
    auto last_node = dfs_convert_nodes(graph, last_node_in_nodes_4d);
    nnfusion::op::OpConfig::any nhwc2nchw_transpose_config;
    nhwc2nchw_transpose_config["axes_order"] = {0, 3, 1, 2};
    auto nhwc2nchw_transpose_op = std::make_shared<nnfusion::op::GenericOp>(
        graph->get_name() + "_nhwc2nchw_transpose", "Transpose", nhwc2nchw_transpose_config);
    NNFUSION_LOG(INFO) << "nchw add recover begin";
    auto nhwc2nchw_gnode = graph->add_node_and_edge(nhwc2nchw_transpose_op, {last_node});
    NNFUSION_LOG(INFO) << "nchw add recover end";
    graph->add_edge(nhwc2nchw_gnode, 0, it, 0);
    NNFUSION_LOG(INFO) << "nchw to nhwc end";
    return true;
    }
