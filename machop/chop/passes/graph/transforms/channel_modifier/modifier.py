from copy import deepcopy
import logging
from torch import nn
from torch.nn import ReLU

from ...utils import (
    get_node_actual_target,
    get_parent_name,
)

logger = logging.getLogger(__name__)

CHANNEL_OP = (
    "linear",
    "relu",
    "batchnorm1d",
)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def instantiate_relu(inplace):
    return ReLU(inplace)

def instantiate_batchnorm(num_features, eps, momentum, affine, track_running_stats):
    return nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
    
def redefine_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)

    if default is None:
        raise ValueError(f"default value must be provided.")

    for node in graph.fx_graph.nodes:
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)

        # get parent operator 
        parent = config.get("parent", None)
        if parent is not None:
            parent_config = main_config.get(parent, default)['config']
        
        new_module = None
        # if the node is a linear layer
        if isinstance(get_node_actual_target(node), nn.Linear):
            if name:
                ori_module = graph.modules[node.target]
                in_features = ori_module.in_features
                out_features = ori_module.out_features

                if name == "output_only":
                    out_features *= config["channel_multiplier"]
                elif name == "both":
                    in_features *= parent_config["channel_multiplier"]
                    out_features *= config["channel_multiplier"]
                elif name == "input_only":
                    in_features *= parent_config["channel_multiplier"]
                # create new module
                new_module = instantiate_linear(in_features, out_features, ori_module.bias)

        # if the node is a ReLU layer            
        elif isinstance(get_node_actual_target(node), ReLU):
            if name:
                ori_module = graph.modules[node.target]
                new_module = instantiate_relu(ori_module.inplace)
                setattr(graph.modules[node.target], "inplace", new_module.inplace)
        
        # if the node is a BatchNorm1d layer
        elif isinstance(get_node_actual_target(node), nn.BatchNorm1d):
            if name:
                ori_module = graph.modules[node.target]
                # new BatchNorm1d with the original parameters
                new_module = instantiate_batchnorm(
                    ori_module.num_features, ori_module.eps, ori_module.momentum, ori_module.affine, 
                    ori_module.track_running_stats)
                parent_name, child_name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], child_name, new_module)  
        
        # if the node is a BatchNorm2d layer
        elif isinstance(get_node_actual_target(node), nn.BatchNorm2d):
            if parent is not None:
                ori_module = graph.modules[node.target]
                num_features = ori_module.num_features
                eps = ori_module.eps
                momentum = ori_module.momentum
                affine  = ori_module.affine
                num_features *= parent_config["channel_multiplier"]
                new_module = nn.BatchNorm2d(num_features, eps, momentum, affine)

        # if the node is a Conv2d layer
        elif isinstance(get_node_actual_target(node), nn.Conv2d):
            if name is not None:
                ori_module = graph.modules[node.target]
                in_channels = ori_module.in_channels
                out_channels = ori_module.out_channels
                if name == "output_only":
                    out_channels *= config["channel_multiplier"]
                elif name == "both":
                    in_channels *= parent_config["channel_multiplier"]
                    out_channels *= config["channel_multiplier"]
                elif name == "input_only":
                    in_channels *= parent_config["channel_multiplier"]
                # create new module
                new_module = nn.Conv2d(in_channels, out_channels, kernel_size=ori_module.kernel_size, stride=ori_module.stride, 
                                       padding=ori_module.padding, dilation=ori_module.dilation,groups=ori_module.groups, 
                                       bias=ori_module.bias is not None, padding_mode=ori_module.padding_mode)

        if new_module is not None:
            parent_name, child_name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], child_name, new_module)

    return graph, {}
