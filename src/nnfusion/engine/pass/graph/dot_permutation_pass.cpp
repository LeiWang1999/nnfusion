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
#define OFFSET6D(x, y, z, w, u, v, ld1, ld2, ld3, ld4, ld5) \
    ((x) * (ld1) + (y) * (ld2) + (z) * (ld3) + (w) * (ld4) + (u) * (ld5) + (v))
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
        if (it->get_op_type() != "Dot" && it->get_op_type() != "BatchMatMul" &&
            it->get_op_type() != "QuantLinear" && it->get_op_type() != "Convolution" &&
            it->get_op_type() != "ImplicitGemm" && it->get_op_type() != "Add")
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
        else if (it->get_op_type() == "ImplicitGemm"){
            NNFUSION_LOG(INFO) << "Replace ImplicitGemm with layoutDot";
            // get the input nodes
            auto input_node = it->get_in_edge(0)->get_src();
            auto weight_node = it->get_in_edge(1)->get_src();
            // insert a NCHW2LayoutCNHW Node
            auto implicit_gemm_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(it->get_op_ptr());
            // get N, C, H, W, P, S, D
            const int N = implicit_gemm_op->localOpConfig.getRoot()["N"];
            // C represents the number of output channels
            const int C = implicit_gemm_op->localOpConfig.getRoot()["C"];
            const int H = implicit_gemm_op->localOpConfig.getRoot()["H"];
            const int W = implicit_gemm_op->localOpConfig.getRoot()["W"];
            const int P = implicit_gemm_op->localOpConfig.getRoot()["P"];
            const int S = implicit_gemm_op->localOpConfig.getRoot()["S"];
            const int D = implicit_gemm_op->localOpConfig.getRoot()["D"];
            NNFUSION_LOG(INFO) << "N: " << N << " C: " << C << " H: " << H << " W: " << W << " P: " << P << " S: " << S << " D: " << D;
            nnfusion::op::OpConfig::any NCHW2LayoutCNHWConfig;
            auto nchw2layoutcnhw_op = std::make_shared<nnfusion::op::GenericOp>(
                "NCHW2LayoutCNHW", "NCHW2LayoutCNHW", NCHW2LayoutCNHWConfig);
            auto nchw2layoutcnhw_gnode = graph->add_node_and_edge(nchw2layoutcnhw_op, GNodeVector{input_node});
            auto edge = it->get_in_edge(0);
            graph->remove_edge(edge);
            graph->add_edge(nchw2layoutcnhw_gnode, 0, it, 0);
            // insert a LayoutImplicitGemm Node
            nnfusion::op::OpConfig::any LayoutImplicitGemmConfig;
            // get N, C, H, W, P, S, D
            LayoutImplicitGemmConfig["N"] = N;
            LayoutImplicitGemmConfig["C"] = C;
            LayoutImplicitGemmConfig["H"] = H;
            LayoutImplicitGemmConfig["W"] = W;
            LayoutImplicitGemmConfig["P"] = P;
            LayoutImplicitGemmConfig["S"] = S;
            LayoutImplicitGemmConfig["D"] = D;
            LayoutImplicitGemmConfig["layout"] = true;

            auto layout_implicit_gemm_op = std::make_shared<nnfusion::op::GenericOp>(
                "LayoutImplicitGemm", "LayoutImplicitGemm", LayoutImplicitGemmConfig);
            NNFUSION_LOG(INFO) << "Create LayoutImplicitGemm Node Done";
            auto layout_implicit_gemm_gnode = graph->add_node_and_edge(
                layout_implicit_gemm_op, GNodeVector{nchw2layoutcnhw_gnode, weight_node});
            NNFUSION_LOG(INFO) << "Add LayoutImplicitGemm Node Done";

            // insert a LayoutCNHW2CNHW Node after LayoutImplicitGemm Node

            nnfusion::op::OpConfig::any LayoutCNHW2CNHWConfig;
            auto layoutcnhw2cnhw_op = std::make_shared<nnfusion::op::GenericOp>(
                "LayoutCNHW2CNHW", "LayoutCNHW2CNHW", LayoutCNHW2CNHWConfig);
            auto layoutcnhw2cnhw_gnode = graph->add_node_and_edge(layoutcnhw2cnhw_op, GNodeVector{layout_implicit_gemm_gnode});

            NNFUSION_LOG(INFO) << "create LayoutCNHW2CNHW Node done";

            for (auto& edge : it->get_out_edges())
            {
                auto dst_node = edge->get_dst();
                auto dst_input = edge->get_dst_input();
                graph->remove_edge(edge);
                graph->add_edge(layoutcnhw2cnhw_gnode, 0, dst_node, dst_input);
            }
            graph->remove_node(it);
            NNFUSION_LOG(INFO) << "Replace ImplicitGemm with layouImplicit done";

            // apply layout transform into weight node.
            auto weight_op = dynamic_pointer_cast<nnfusion::op::Constant>(weight_node->get_op_ptr());
            // assert weight_op != nullptr
            NNFUSION_CHECK_NOT_NULLPTR(weight_op);
            auto weight_shape = weight_node->get_output_shape(0);
            NNFUSION_LOG(INFO) << "weight shape is " << weight_shape[0] << " " << weight_shape[1] << " " << weight_shape[2] << " " << weight_shape[3];
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
                // assume weight is nchw-> nhwc
                auto N = weight_shape[0];
                auto C = weight_shape[1];
                auto H = weight_shape[2];
                auto W = weight_shape[3];
                NNFUSION_CHECK(N % 16 == 0);
                NNFUSION_CHECK(C % 16 == 0);

                // permutate nchw with (n // 16, c // 16, h, w, n % 16, c % 16)
                for (int n = 0; n < N / 16; n++)
                    for (int c = 0; c < C / 16; c++)
                        for (int h = 0; h < H; h++)
                            for (int w = 0; w < W; w++)
                                for (int nn = 0; nn < 16; nn++)
                                    for (int cc = 0; cc < 16; cc++)
                                    {
                                        temp_data[OFFSET6D(n,
                                                           c,
                                                           h,
                                                           w,
                                                           nn,
                                                           cc,
                                                           H * W * C * 16,
                                                           W * H * 256,
                                                           W * 256,
                                                           256,
                                                           16)] =
                                            data[OFFSET4D(n * 16 + nn , c * 16 + cc, h, w, H * W * C, W * H, W)];
                                    }
                // cp temp_data to data
                memcpy(data, temp_data, weight_op->get_data_size());
            }
            else{
                NNFUSION_LOG(ERROR) << "weight_node's element_type is not f16";
            }
        }else if (it->get_op_type() == "Convolution"){
            NNFUSION_LOG(INFO) << "Transform Convolution Data with LayoutTransform";

            auto conv_op = static_pointer_cast<nnfusion::op::Convolution>(it->get_op_ptr());
            auto input_node = it->get_in_edge(0)->get_src();
            // get input channel
            auto input_shape = input_node->get_output_shape(0);
            auto input_channel = conv_op->get_data_format() == "NCHW" ? input_shape[1] : input_shape[3];
            if (input_channel % 16 != 0)
            {
                NNFUSION_LOG(INFO) << "input channel is not multiple of 16, skip";
                continue;
            }

            nnfusion::op::OpConfig::any NHWC2LayoutNHWCConfig;
            auto nhwc2layoutnhwc_op = std::make_shared<nnfusion::op::GenericOp>(
                "NHWC2LayoutNHWC", "NHWC2LayoutNHWC", NHWC2LayoutNHWCConfig);
            auto nhwc2layoutnhwc_gnode = graph->add_node_and_edge(nhwc2layoutnhwc_op, GNodeVector{input_node});
            auto edge = it->get_in_edge(0);
            graph->remove_edge(edge);
            graph->add_edge(nhwc2layoutnhwc_gnode, 0, it, 0);
            NNFUSION_LOG(INFO) << "create LayoutCNHW2CNHW Node done";
            auto weight_node = it->get_in_edge(1)->get_src();
            auto weight_op =
                dynamic_pointer_cast<nnfusion::op::Constant>(weight_node->get_op_ptr());
            // assert weight_op != nullptr
            NNFUSION_CHECK_NOT_NULLPTR(weight_op);
            auto weight_shape = weight_node->get_output_shape(0);
            NNFUSION_LOG(INFO) << "weight shape is " << weight_shape[0] << " " << weight_shape[1]
                               << " " << weight_shape[2] << " " << weight_shape[3];
            // get element_type
            auto element_type = weight_op->get_type();

            if (element_type == nnfusion::element::f16)
            {
                NNFUSION_LOG(INFO) << "weight_node's element_type is f16";
                NNFUSION_CHECK(conv_op->get_data_format() == "NHWC") << "conv_op's data format is not NHWC, currently not support";
                // rewrite data as first transpose
                // get data
                half_float::half* data = (half_float::half*)weight_op->get_data_ptr();
                // create a temp storage
                half_float::half* temp_data =
                    (half_float::half*)(new char[weight_op->get_data_size()]);
                // assume weight is nchw-> nhwc
                auto N = weight_shape[0];
                auto H = weight_shape[1];
                auto W = weight_shape[2];
                auto C = weight_shape[3];
                NNFUSION_CHECK(N % 16 == 0);
                NNFUSION_CHECK(C % 16 == 0);

                // permutate nchw with (n // 16, c // 16, h, w, n % 16, c % 16)
                for (int n = 0; n < N / 16; n++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            for (int c = 0; c < C / 16; c++)
                                for (int nn = 0; nn < 16; nn++)
                                    for (int cc = 0; cc < 16; cc++)
                                    {
                                        temp_data[OFFSET6D(n,
                                                           h,
                                                           w,
                                                           c,
                                                           nn,
                                                           cc,
                                                           H * W * C * 16,
                                                           W * C * 16,
                                                           C * 16,
                                                           256,
                                                           16)] = data[OFFSET4D(n * 16 + nn,
                                                                                h,
                                                                                w,
                                                                                c * 16 + cc,
                                                                                H * W * C,
                                                                                W * C,
                                                                                C)];
                                    }
                // cp temp_data to data
                memcpy(data, temp_data, weight_op->get_data_size());
            }
    }
        else if (it->get_op_type() == "Add"){
            NNFUSION_LOG(INFO) << "Transform Add Data with LayoutTransform";
            auto add_op = static_pointer_cast<nnfusion::op::Add>(it->get_op_ptr());
            auto input0_node = it->get_in_edge(0)->get_src();
            auto input1_node = it->get_in_edge(1)->get_src();
            auto output_edgs = it->get_out_edges();
            NNFUSION_LOG(INFO) << "output edge size is " << output_edgs.size();
            // filter rule: one of the input node is a bias add node
            if(input1_node->get_op_type() != "Add" && input0_node->get_op_type() != "Add"){
                NNFUSION_LOG(INFO) << "input node is not bias add node, skip";
                continue;
            }else{
                NNFUSION_LOG(INFO) << "input node is bias add node, transform";
                // get which node is bias add node
                auto bias_add_node = input1_node->get_op_type() == "Add" ? input1_node : input0_node;
                NNFUSION_LOG(INFO) << "bias add node is " << bias_add_node->get_op_type();
                // get the node beside bias add node
                auto beside_node = input1_node->get_op_type() == "Add" ? input0_node : input1_node;
                auto output_edgs = beside_node->get_out_edges();
                NNFUSION_LOG(INFO) << "beside_node output edge size is " << output_edgs.size();
                NNFUSION_LOG(INFO) << "beside node is " << beside_node->get_op_type();
                auto beside_id = input1_node->get_op_type() == "Add" ? 0 : 1;
                // insert a NHWC2LayoutNHWC node
                nnfusion::op::OpConfig::any NHWC2LayoutNHWCConfig;
                auto nhwc2layoutnhwc_op = std::make_shared<nnfusion::op::GenericOp>(
                    "NHWC2LayoutNHWC", "NHWC2LayoutNHWC", NHWC2LayoutNHWCConfig);
                auto nhwc2layoutnhwc_gnode = graph->add_node_and_edge(nhwc2layoutnhwc_op, GNodeVector{beside_node});
                auto edge = it->get_in_edge(beside_id);
                graph->remove_edge(edge);
                graph->add_edge(nhwc2layoutnhwc_gnode, 0, it, beside_id);
                NNFUSION_LOG(INFO) << "create LayoutCNHW2CNHW Node done";
            }
        }
    }
#undef OFFSET2D
#undef OFFSET4D
#undef OFFSET6D
    // handle the case that two NHWC2LayoutNHWC node are connected
    for (auto& it : nodes)
    {
        if(it->get_op_type() == "Add"){
            auto add_op = static_pointer_cast<nnfusion::op::Add>(it->get_op_ptr());
            auto input0_node = it->get_in_edge(0)->get_src();
            auto input1_node = it->get_in_edge(1)->get_src();
            if (input0_node->get_op_type() == "Convolution" &&
                input1_node->get_op_type() == "Convolution"){
                    NNFUSION_LOG(INFO) << "merge two NHWC2LayoutNHWC into one";
            }
        }
    }
    // simplify graph merge two NHWC2LayoutNHWC into one
    for (auto& it : nodes)
    {
        auto output_edges = it->get_out_edges();
        if(output_edges.size() == 2){
            NNFUSION_LOG(INFO) << it->get_name() <<" output edge size is 2";
            auto edge0 = output_edges[0];
            auto edge1 = output_edges[1];
            auto dst0 = edge0->get_dst();
            auto dst1 = edge1->get_dst();
            if (dst0->get_op_type() == "NHWC2LayoutNHWC" &&
                dst1->get_op_type() == "NHWC2LayoutNHWC"){
                NNFUSION_LOG(INFO) << "merge two NHWC2LayoutNHWC into one";
                // keep dst0 node, remove dst1 node, connect the dst of dst1 to dst0
                auto dst1_output_edges = dst1->get_out_edges();
                auto dst1_output_node = dst1_output_edges[0]->get_dst();
                auto dst1_output_input = dst1_output_edges[0]->get_dst_input();
                NNFUSION_LOG(INFO) << "dst1_output_node is " << dst1_output_node->get_op_type();
                NNFUSION_LOG(INFO) << "dst1_output_input is " << dst1_output_input;
                graph->remove_node(dst1);
                graph->add_edge(dst0, 0, dst1_output_node, dst1_output_input);
            }
        }
    }
    return true;
}
