// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

static string make_layout(const std::vector<int>& axes)
{
    std::string ret = "";
    for (auto ax : axes)
        ret += ", N" + std::to_string(ax);
    return "[" + (axes.empty() ? "N" : ret.substr(2)) + "]";
};

REGISTER_OP(Transpose)
    .attr<std::vector<int>>("axes_order")
    .infershape(
        [](std::shared_ptr<graph::GNode> gnode) -> void
        {
            auto& shape_0 = gnode->get_input_shape(0);
            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
            auto& axes_order = generic_op->localOpConfig.getRoot()["axes_order"];
            NNFUSION_CHECK(axes_order.size() == shape_0.size());
            nnfusion::Shape output_shape_0;
            for (int i = 0; i < axes_order.size(); ++i)
                output_shape_0.push_back(shape_0[axes_order[i]]);
            gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
        })
    .translate_v2(
        [](std::shared_ptr<graph::GNode> curr) -> std::string
        {
            auto input_shape = curr->get_input_shape(0);
            auto generic_op =
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
            auto& axes_order = generic_op->localOpConfig.getRoot()["axes_order"];

            std::vector<int> input_ax, output_ax;
            for (int i = 0; i < input_shape.size(); ++i)
                input_ax.emplace_back(i);
            for (int i = 0; i < axes_order.size(); ++i)
                output_ax.emplace_back(axes_order[i]);

            string expression_template =
                R"( @output0@@output0_layout@ = @input0@@input0_layout@; )";
            string expression_code =
                op::create_code_from_template(expression_template,
                                              {{"output0_layout", make_layout(output_ax)},
                                               {"input0_layout", make_layout(input_ax)}});
            return expression_code;
        });
