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

def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]
    
def redefine_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)

    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    # pre_in = 1
    # pre_out = 1

    # Print the original graph
    # print("Original Graph:")
    # for block in graph.model.seq_blocks._modules:
    #     print(f"Module number {block}: {graph.model.seq_blocks._modules[block]}")


    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        
        actual_target = get_node_actual_target(node)
        new_module = None
        if isinstance(actual_target, nn.Linear):
            if name is not None:
                if node.target=='x' or node.target=='output':
                    continue
                
                ori_module = graph.modules[node.target]
                in_features = ori_module.in_features
                out_features = ori_module.out_features
                # in_features = config.get('in_features', 16)
                # out_features = config.get('out_features', 16)
                bias = ori_module.bias
                if name == "output_only":
                    # in_features = ori_module.in_features
                    out_features = out_features * config["channel_multiplier"]
                    # pre_out=config["channel_multiplier"]
                elif name == "both":
                    # in_features = in_features * pre_out
                    in_features = in_features * main_config.get(config['parent'], default)['config']["channel_multiplier"]
                    out_features = out_features * config["channel_multiplier"]
                    # pre_out = pre_in
                    # pre_in = config["channel_multiplier"]
                elif name == "input_only":
                    # in_features = in_features * pre_in
                    in_features = in_features * main_config.get(config['parent'], default)['config']["channel_multiplier"]
                    # out_features = ori_module.out_features
                new_module = instantiate_linear(in_features, out_features, bias)
                # parent_name, name = get_parent_name(node.target)
                # setattr(graph.modules[parent_name], name, new_module)
            
        elif isinstance(actual_target, ReLU):
            name = config.get("name")
            if name:
                ori_module = graph.modules[node.target]
                new_module = instantiate_relu(ori_module.inplace)
                setattr(graph.modules[node.target], "inplace", new_module.inplace)
        
        elif isinstance(actual_target, nn.BatchNorm1d):
            name = config.get("name")
            if name:
                ori_module = graph.modules[node.target]
                # new BatchNorm1d with the original parameters
                new_module = instantiate_batchnorm(
                    ori_module.num_features, ori_module.eps, ori_module.momentum, 
                    ori_module.affine, ori_module.track_running_stats)
                parent_name, child_name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], child_name, new_module)  
                
        elif isinstance(actual_target, nn.BatchNorm2d):
            parent = config.get("parent", None)
            if parent is not None:
                ori_module = graph.modules[node.target]
                num_features, eps, momentum, affine = ori_module.num_features, ori_module.eps, ori_module.momentum, ori_module.affine
                num_features = num_features * main_config.get(parent, default)['config']["channel_multiplier"]
                new_module = nn.BatchNorm2d(num_features, eps, momentum, affine)

        elif isinstance(actual_target, nn.Conv2d):
            # name = config.get("name", None)
            if name is not None:
                ori_module = graph.modules[node.target]
                in_channels = ori_module.in_channels
                out_channels = ori_module.out_channels
                bias = ori_module.bias
                if name == "output_only":
                    out_channels = out_channels * config["channel_multiplier"]
                elif name == "both":
                    in_channels = in_channels * main_config.get(config['parent'], default)['config']["channel_multiplier"]
                    out_channels = out_channels * config["channel_multiplier"]
                elif name == "input_only":
                    in_channels = in_channels * main_config.get(config['parent'], default)['config']["channel_multiplier"]
                new_module = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=ori_module.kernel_size, stride=ori_module.stride,
                                       padding=ori_module.padding, dilation=ori_module.dilation,
                                       groups=ori_module.groups, bias=ori_module.bias is not None,
                                       padding_mode=ori_module.padding_mode)

        if new_module is not None:
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)

        # Print the transformed graph
        # print("Transformed Graph:")
        # for block in graph.model.seq_blocks._modules:
        #     print(f"Module number {block}: {graph.model.seq_blocks._modules[block]}")

    return graph, {}
