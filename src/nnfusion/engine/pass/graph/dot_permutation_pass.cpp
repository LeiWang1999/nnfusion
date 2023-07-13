#include "dot_permutation_pass.hpp"
#include "kernel_selection.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/graph/graph_util.hpp"
#include "nnfusion/core/operators/op_define/broadcast.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"
#include "nnfusion/util/util.hpp"
#include "gflags/gflags.h"

#include <queue>

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(fdot_permutation, false, "Enable Dot Permutation Pass");
DEFINE_string(fpermutate_skiplist, "", "List of op types that skips in permutation");
DEFINE_string(fsplitk_factorlist, "", "List of split k factors in ops");

namespace{

    std::unordered_set<std::string> skip_ops = {};
    void parse_skip_ops()
    {
        stringstream ss(FLAGS_fpermutate_skiplist);
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            skip_ops.insert(substr);
        }
    }
    std::unordered_map<std::string, int> splitk_factors = {};
    void parse_splitk_factors()
    {
        /*
            m128n128k128:1;
        */
        stringstream ss(FLAGS_fsplitk_factorlist);
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ';');
            if (substr.empty())
                continue;
            auto pos = substr.find(':');
            if (pos == std::string::npos)
                continue;
            std::string key = substr.substr(0, pos);
            std::string value = substr.substr(pos + 1);
            int factor = std::stoi(value);
            splitk_factors[key] = factor;
        }
    }
}

bool DotPermutationPass::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    bool using_pass = FLAGS_fdot_permutation;
    if (!using_pass)
        return true;
#define OFFSET2D(x, y, ld) ((x) * (ld) + (y))
#define OFFSET4D(x, y, z, w, ld1, ld2, ld3) ((x) * (ld1) + (y) * (ld2) + (z) * (ld3) + (w))
    parse_skip_ops();
    parse_splitk_factors();
    // print splitk_factors
    for (auto& it : splitk_factors)
    {
        NNFUSION_LOG(INFO) << "matrix info: " << it.first << " splitk factor: " << it.second;
    }
    NNFUSION_LOG(INFO) << "DotPermutationPass::run_on_graph start";
    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    size_t kernel_i = 16;
    size_t kernel_j = 16;
    size_t kernel_k = 16;
    for (auto& it : nodes)
    {
        if (skip_ops.count(it->get_op_type()))
            continue;
        if (it->get_op_type() != "Dot" && it->get_op_type() != "BatchMatMul" && it->get_op_type() != "QuantLinear")
        {
            continue;
        }
        if (it->get_op_type() == "Dot"){
            
            // find a dot node
            NNFUSION_LOG(INFO) << "Find a dot node: " << it->get_id();  
            auto it_op = static_pointer_cast<nnfusion::op::Dot>(it->get_op_ptr());
            auto trans_a = it_op->get_transpose_A();
            auto trans_b = it_op->get_transpose_B();
            NNFUSION_LOG(INFO) << "trans_a " << trans_a << " trans_b " << trans_b;
            // get the input nodes
            auto input_node = it->get_in_edge(0)->get_src();
            auto weight_node = it->get_in_edge(1)->get_src();
            NNFUSION_LOG(INFO) << "Input node: " << input_node->get_id();
            NNFUSION_LOG(INFO) << "Input Type: " << input_node->get_unique_name();
            NNFUSION_LOG(INFO) << "Weight node: " << weight_node->get_id();
            // if the input_node or weight_node is dot, continue
            if (input_node->get_op_type() == "Dot" || weight_node->get_op_type() == "Dot")
                NNFUSION_LOG(ERROR) << "Currently do not support input node or weight node is dot";
            
            // create a new Permutate Node;
            nnfusion::op::OpConfig::any permutateConfig;
            permutateConfig["type"] = 0;
            permutateConfig["inner_i"] = kernel_i;
            permutateConfig["inner_j"] = kernel_k;
            auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                "Permutate", "Permutate", permutateConfig);
            // convert op to GNode
            auto permutate_node = graph->add_node_and_edge(generic_op, {input_node});
            auto edge = it->get_in_edge(0);
            graph->remove_edge(edge);
            graph->add_edge(permutate_node, 0, it, 0);
            // replace dot with LayoutDot
            nnfusion::op::OpConfig::any layoutDotConfig;
            layoutDotConfig["output_type"] = 0;
            layoutDotConfig["inner_i"] = kernel_i;
            layoutDotConfig["inner_j"] = kernel_j;
            layoutDotConfig["transpose_A"] = trans_a;
            layoutDotConfig["transpose_B"] = trans_b;
            // get key m128n128k128
            // get dot shape m n k
            auto input_shape = input_node->get_output_shape(0);
            auto weight_shape = weight_node->get_output_shape(0);
            size_t k =
                trans_a ? input_shape[input_shape.size() - 2] : input_shape[input_shape.size() - 1];
            // get M=batch*m since input_shape is [batch, m, k] or [batch, k, m] or just [m, k]
            // print input_shape
            for (auto& it : input_shape)
            {
                NNFUSION_LOG(INFO) << "input_shape: " << it;
            }
            size_t m = 1;
            if (3 == input_shape.size()) {
                m = input_shape[0];
                m = m * (trans_a ? input_shape[input_shape.size() - 1]
                                 : input_shape[input_shape.size() - 2]);
            }else if (2 == input_shape.size()) {
                m = trans_a ? input_shape[input_shape.size() - 1]
                            : input_shape[input_shape.size() - 2];
            }else{
                NNFUSION_LOG(ERROR) << "input shape is not supported";
            }
            size_t n = trans_b ? weight_shape[weight_shape.size() - 2]
                               : weight_shape[weight_shape.size() - 1];
            std::string key = "m" + std::to_string(m) + "n" + std::to_string(n) + "k" + std::to_string(k);
            NNFUSION_LOG(INFO) << "key is " << key;
            // get splitk factor
            if (splitk_factors.count(key) != 0){
                NNFUSION_LOG(INFO) << "splitk_factors[" << key << "] is " << splitk_factors[key];
                layoutDotConfig["splitk"] = splitk_factors[key];
            }
            auto layoutDot_op = std::make_shared<nnfusion::op::GenericOp>(
                "LayoutDot", "LayoutDot", layoutDotConfig);
            NNFUSION_LOG(INFO) << "Create layoutDot node";
            // add layoutDot_node behind layout_Dot_op
            NNFUSION_LOG(INFO) << "permutate_node input shape is " << nnfusion::join(permutate_node->get_input_shape(0));
            NNFUSION_LOG(INFO) << "permutate_node shape is "
                            << nnfusion::join(permutate_node->get_shape());

            auto layoutDot_node = graph->add_node_and_edge(layoutDot_op, {permutate_node, weight_node});
            NNFUSION_LOG(INFO) << "Replace it->output's input edge with layoutDot_node";
            if (splitk_factors.count(key) != 0){
                // if split, insert a reduce sum node after layoutDot_node
                auto splitk_factor = splitk_factors[key];
                auto reduction_axes = nnfusion::AxisSet({0});
                auto sum_node = std::make_shared<op::Sum>(reduction_axes);
                auto sum_gnode = graph->add_node_and_edge(sum_node, {layoutDot_node});
                for (auto& edge : it->get_out_edges())
                {
                    auto dst_node = edge->get_dst();
                    auto dst_input = edge->get_dst_input();
                    graph->remove_edge(edge);
                    graph->add_edge(sum_gnode, 0, dst_node, dst_input);
                }
            }else{
                for (auto& edge : it->get_out_edges())
                {
                    auto dst_node = edge->get_dst();
                    auto dst_input = edge->get_dst_input();
                    graph->remove_edge(edge);
                    graph->add_edge(layoutDot_node, 0, dst_node, dst_input);
                }
            }

            graph->remove_node(it);
            NNFUSION_LOG(INFO) << "Replace dot with layoutDot done";
            // apply layout transform into weight_node
            NNFUSION_LOG(INFO) << "Apply layout transform into weight_node";
            auto weight_op = dynamic_pointer_cast<nnfusion::op::Constant>(weight_node->get_op_ptr());
            // assert weight_op != nullptr
            NNFUSION_CHECK_NOT_NULLPTR(weight_op);
            NNFUSION_LOG(INFO) << "weight shape is " << weight_shape[0] << " " << weight_shape[1];
            // get element_type
            auto element_type = weight_op->get_type();

            if (element_type == nnfusion::element::f16)
            {
                NNFUSION_LOG(INFO) << "weight_node's element_type is f16";
                // rewrite data as first transpose
                // get data
                half_float::half* data = (half_float::half *)weight_op->get_data_ptr();
                // create a temp storage
                half_float::half* temp_data = (half_float::half*)(new char[weight_op->get_data_size()]);
                // transpose
                // if weight is transposed, direct assign
                if (it_op->get_transpose_B())
                {
                    NNFUSION_LOG(INFO) << "weight_node is transposed";
                    memcpy(temp_data, data, weight_op->get_data_size());
                }
                else
                {
                    NNFUSION_LOG(INFO) << "weight_node is not transposed";

                    for (int i = 0; i < weight_shape[0]; i++)
                    {
                        for (int j = 0; j < weight_shape[1]; j++)
                        {
                            temp_data[OFFSET2D(j, i, weight_shape[0])] =
                                data[OFFSET2D(i, j, weight_shape[1])];
                        }
                    }
                }


                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        data[OFFSET4D(i / 16,
                                    j / 16,
                                    i % 16,
                                    j % 16,
                                    kernel_j * k,
                                    kernel_j * kernel_k,
                                    kernel_k)] =
                            temp_data[OFFSET2D(i / 8 * 8 + i % 4 * 2 + j % 16 / 8,
                                            j / 16 * 16 + i % 8 / 4 * 8 + j % 8,
                                            k)];
                    }
                }
            }
            else{
                NNFUSION_LOG(ERROR) << "weight_node's element_type is not f16";
            }
        }
        else if (it->get_op_type() == "BatchMatMul"){
            NNFUSION_LOG(INFO) << "Find a BatchMatMul node: " << it->get_id();
            // get the input nodes
            auto input_node = it->get_in_edge(0)->get_src();
            auto weight_node = it->get_in_edge(1)->get_src();
            NNFUSION_LOG(INFO) << "Input node: " << input_node->get_id();
            NNFUSION_LOG(INFO) << "Input Type: " << input_node->get_unique_name();
            NNFUSION_LOG(INFO) << "Weight node: " << weight_node->get_id();
            // get node's attr
            auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(it->get_op_ptr());
            bool trans_A = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
            bool trans_B = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

            // Currently do not support constant weight
            NNFUSION_CHECK(weight_node->get_op_type() != "Constant") << "Constant weight is not supported for now";
            // Insert permutate node before BatchMatMul's input node and weight node
            // Insert permutate node before BatchMatMul's input node
            NNFUSION_LOG(INFO) << "Insert permutate node before BatchMatMul's input node";
            {
                // permutate input
                nnfusion::op::OpConfig::any permutateConfig;
                permutateConfig["type"] = trans_A? 1 : 0;
                permutateConfig["inner_i"] = kernel_i;
                permutateConfig["inner_j"] = kernel_k;
                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    "BatchPermutate", "BatchPermutate", permutateConfig);
                auto permutate_node = graph->add_node_and_edge(generic_op, {input_node});
                auto edge = it->get_in_edge(0);
                graph->remove_edge(edge);
                graph->add_edge(permutate_node, 0, it, 0);
            }
            {
                // permutate weight
                nnfusion::op::OpConfig::any permutateConfig;
                permutateConfig["type"] = trans_B ? 1 : 0;
                permutateConfig["inner_i"] = kernel_i;
                permutateConfig["inner_j"] = kernel_k;
                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    "BatchPermutate", "BatchPermutate", permutateConfig);
                auto permutate_node = graph->add_node_and_edge(generic_op, {weight_node});
                auto edge = it->get_in_edge(1);
                graph->remove_edge(edge);
                graph->add_edge(permutate_node, 0, it, 1);
            }
            // replace BatchMatMul with LayoutBMM
            NNFUSION_LOG(INFO) << "Replace BatchMatMul with LayoutBMM";
            {
                nnfusion::op::OpConfig::any layoutBMMConfig;
                layoutBMMConfig["output_type"] = 0;
                layoutBMMConfig["inner_i"] = kernel_i;
                layoutBMMConfig["inner_j"] = kernel_j;
                auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                    "LayoutBMM", "LayoutBMM", layoutBMMConfig);
                NNFUSION_LOG(INFO) << "Create LayoutBMM node";
                auto layoutBMM_node = graph->add_node_and_edge(
                    generic_op, {it->get_in_edge(0)->get_src(), it->get_in_edge(1)->get_src()});
                for (auto& edge : it->get_out_edges())
                {
                    auto dst_node = edge->get_dst();
                    auto dst_input = edge->get_dst_input();
                    graph->remove_edge(edge);
                    graph->add_edge(layoutBMM_node, 0, dst_node, dst_input);
                }
                graph->remove_node(it);
            }
        }
        else if(it->get_op_type() == "QuantLinear"){
            // find a dot node
            NNFUSION_LOG(INFO) << "Find a QuantLinear node: " << it->get_id();
            auto quant_linear_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(it->get_op_ptr());
            bool trans_a = quant_linear_op->localOpConfig.getRoot()["transpose_A"];
            bool trans_b = quant_linear_op->localOpConfig.getRoot()["transpose_B"];
            int bits = quant_linear_op->localOpConfig.getRoot()["bits"];
            NNFUSION_LOG(INFO) << "trans_a " << trans_a << " trans_b " << trans_b << " bits "
                               << bits;
            auto input_node = it->get_in_edge(0)->get_src();
            auto qweight_node = it->get_in_edge(1)->get_src();
            auto scale_node = it->get_in_edge(2)->get_src();
            auto zero_point_node = it->get_in_edge(3)->get_src();
            NNFUSION_LOG(INFO) << "Input node: " << input_node->get_id();
            NNFUSION_LOG(INFO) << "Input Type: " << input_node->get_unique_name();
            NNFUSION_LOG(INFO) << "QWeight node: " << qweight_node->get_id();
            // if the input_node or weight_node is dot, continue
            if (input_node->get_op_type() == "QuantLinear" ||
                qweight_node->get_op_type() == "QuantLinear")
                NNFUSION_LOG(ERROR)
                    << "Currently do not support input node or weight node is QuantLinear";
            // create a new Permutate Node;
            nnfusion::op::OpConfig::any permutateConfig;
            permutateConfig["type"] = 0;
            permutateConfig["inner_i"] = kernel_i;
            permutateConfig["inner_j"] = kernel_k;
            auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                "Permutate", "Permutate", permutateConfig);
            // convert op to GNode
            auto permutate_node = graph->add_node_and_edge(generic_op, {input_node});
            auto edge = it->get_in_edge(0);
            graph->remove_edge(edge);
            graph->add_edge(permutate_node, 0, it, 0);
            // replace QuantLinear with LayoutQuantLinear
            nnfusion::op::OpConfig::any layoutQuantLinearConfig;
            layoutQuantLinearConfig["output_type"] = 0;
            layoutQuantLinearConfig["inner_i"] = kernel_i;
            layoutQuantLinearConfig["inner_j"] = kernel_j;
            layoutQuantLinearConfig["transpose_A"] = trans_a;
            layoutQuantLinearConfig["transpose_B"] = trans_b;
            // get key m128n128k128
            // get dot shape m n k
            auto input_shape = input_node->get_output_shape(0);
            auto weight_shape = qweight_node->get_output_shape(0);
            size_t k =
                trans_a ? input_shape[input_shape.size() - 2] : input_shape[input_shape.size() - 1];
            // get M=batch*m since input_shape is [batch, m, k] or [batch, k, m] or just [m, k]
            // print input_shape
            for (auto& it : input_shape)
            {
                NNFUSION_LOG(INFO) << "input_shape: " << it;
            }
            size_t m = 1;
            if (3 == input_shape.size())
            {
                m = input_shape[0];
                m = m * (trans_a ? input_shape[input_shape.size() - 1]
                                 : input_shape[input_shape.size() - 2]);
            }
            else if (2 == input_shape.size())
            {
                m = trans_a ? input_shape[input_shape.size() - 1]
                            : input_shape[input_shape.size() - 2];
            }
            else
            {
                NNFUSION_LOG(ERROR) << "input shape is not supported";
            }
            size_t n = trans_b ? weight_shape[weight_shape.size() - 2]
                               : weight_shape[weight_shape.size() - 1];
            std::string key =
                "m" + std::to_string(m) + "n" + std::to_string(n) + "k" + std::to_string(k);
            NNFUSION_LOG(INFO) << "key is " << key;
            // get splitk factor
            if (splitk_factors.count(key) != 0)
            {
                NNFUSION_LOG(INFO) << "splitk_factors[" << key << "] is " << splitk_factors[key];
                layoutQuantLinearConfig["splitk"] = splitk_factors[key];
            }
            auto layoutQuantLinear_op = std::make_shared<nnfusion::op::GenericOp>(
                "layoutQuantLinear", "layoutQuantLinear", layoutQuantLinearConfig);
            NNFUSION_LOG(INFO) << "Create layoutDot node";
            // add layoutDot_node behind layout_Dot_op
            NNFUSION_LOG(INFO) << "permutate_node input shape is "
                               << nnfusion::join(permutate_node->get_input_shape(0));
            NNFUSION_LOG(INFO) << "permutate_node shape is "
                               << nnfusion::join(permutate_node->get_shape());
            auto layoutQuantLinear_node = graph->add_node_and_edge(
                layoutQuantLinear_op, {permutate_node, qweight_node, scale_node, zero_point_node});
            if (splitk_factors.count(key) != 0)
            {
                // if split, insert a reduce sum node after layoutDot_node
                auto splitk_factor = splitk_factors[key];
                auto reduction_axes = nnfusion::AxisSet({0});
                auto sum_node = std::make_shared<op::Sum>(reduction_axes);
                auto sum_gnode = graph->add_node_and_edge(sum_node, {layoutQuantLinear_node});
                for (auto& edge : it->get_out_edges())
                {
                    auto dst_node = edge->get_dst();
                    auto dst_input = edge->get_dst_input();
                    graph->remove_edge(edge);
                    graph->add_edge(sum_gnode, 0, dst_node, dst_input);
                }
            }
            else
            {
                for (auto& edge : it->get_out_edges())
                {
                    auto dst_node = edge->get_dst();
                    auto dst_input = edge->get_dst_input();
                    graph->remove_edge(edge);
                    graph->add_edge(layoutQuantLinear_node, 0, dst_node, dst_input);
                }
            }
            graph->remove_node(it);
            NNFUSION_LOG(INFO) << "Replace dot with layoutDot done";
            // apply layout transform into weight_node
            NNFUSION_LOG(INFO) << "Apply layout transform into weight_node";
            auto weight_op =
                dynamic_pointer_cast<nnfusion::op::Constant>(qweight_node->get_op_ptr());
            // assert weight_op != nullptr
            NNFUSION_CHECK_NOT_NULLPTR(weight_op);
            NNFUSION_LOG(INFO) << "weight shape is " << weight_shape[0] << " " << weight_shape[1];
            // get element_type
            auto element_type = weight_op->get_type();
            if (element_type == nnfusion::element::i8)
            {
                NNFUSION_LOG(INFO) << "weight_node's element_type is f16";
                // rewrite data as first transpose
                // get data
                int8_t* data = (int8_t*)weight_op->get_data_ptr();
                // create a temp storage
                int8_t* temp_data = (int8_t*)(new char[weight_op->get_data_size()]);
                // transpose
                // if weight is transposed, direct assign
                if (trans_b)
                {
                    NNFUSION_LOG(INFO) << "weight_node is transposed";
                    memcpy(temp_data, data, weight_op->get_data_size());
                }
                else
                {
                    NNFUSION_LOG(INFO) << "weight_node is not transposed";

                    for (int i = 0; i < weight_shape[0]; i++)
                    {
                        for (int j = 0; j < weight_shape[1]; j++)
                        {
                            temp_data[OFFSET2D(j, i, weight_shape[0])] =
                                data[OFFSET2D(i, j, weight_shape[1])];
                        }
                    }
                }
                /*
                    recover data from int8 to int4 with the following algo
                        mask = (1 << bit) - 1
                        b_int8 = np.zeros((N, K)).astype("int8")
                        for vi in range(N):
                            for vj in range(K):
                                # get int8 from compressed int4
                                b_int8[vi, vj] = b_compressed[vi, vj // 2] >> (vj % 2 * 4) & mask
                */
                int bit = 4; // todo(v-leiwang3): get bit from quantization layer attr
                assert(bit == 4);

                int mask = (1 << bit) - 1;
                int8_t* b_int8 = (int8_t*)(new char[weight_op->get_data_size() / bit * 8]);
                for (int vi = 0; vi < n; vi++)
                {
                    for (int vj = 0; vj < k; vj++)
                    {
                        b_int8[OFFSET2D(vi, vj, k)] =
                            temp_data[OFFSET2D(vi, vj / 2, k / 2)] >>
                                (vj % 2 * 4) & mask;
                    }
                }
                int8_t* b_int8_prmt = (int8_t*)(new char[weight_op->get_data_size() / bit * 8]);
                // layout transform data[vi / 16, vj / 16, vi % 16, vj % 16] = temp_data[vi / 8 * 8 + vi % 4 * 2 + vj % 16 / 8, vj / 16 * 16 + vi % 8 / 4 * 8 + vj % 8
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        b_int8_prmt[OFFSET2D(i, j, k)] =
                            b_int8[OFFSET2D(i / 8 * 8 + i % 4 * 2 + j % 16 / 8,
                                            j / 16 * 16 + i % 8 / 4 * 8 + j % 8,
                                            k)];
                    }
                }
                // compressed data
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        temp_data[OFFSET2D(i, j / 2, k)] |= b_int8_prmt[OFFSET2D(i, j, k)]
                                                             << (j % 2 * 4);
                    }
                }
                // transform data

                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < (k / 2); j++)
                    {
                        data[OFFSET4D(i / 16,
                                      j / 8,
                                      i % 16,
                                      j % 8,
                                      kernel_j * k,
                                      kernel_j * kernel_k,
                                      kernel_k)] =
                            temp_data[OFFSET2D(i, j, k)];
                    }
                }
            }
            else
            {
                NNFUSION_LOG(ERROR) << "weight_node's element_type is not f16";
            }
        }
    }
#undef OFFSET2D
#undef OFFSET4D
    return true;
}
