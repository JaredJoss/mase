import torch
from chop.passes.graph.utils import get_node_actual_target

def count_flops_mg_analysis_pass(graph, pass_args: dict):
    """
    Calculates the number of floating-point operations (FLOPs) on the given graph.

    :param graph: The graph to analyze.
    :type graph: MaseGraph
    :param pass_args: Additional arguments for the analysis pass.
    :type pass_args: dict

    :return: A tuple containing the analyzed graph and a dictionary with the module statistics and calculated FLOPs.
    :rtype: tuple
    :return graph: The analyzed graph.
    :rtype graph: MaseGraph
    :return dict: A dictionary with the following keys:
        - 'modules_dict' (dict): A dictionary of module statistics.
        - 'flops' (int): The number of FLOPs in a graph.
    :rtype dict: dict
    """
        
    modules_dict = {}
    flops = 0
    for node in graph.fx_graph.nodes:
        try:
            data_in = (node.meta['mase'].parameters['common']['args']['data_in_0']['value'],)
        except KeyError:
            data_in = (None,)
        data_out = (node.meta['mase'].parameters['common']['results']['data_out_0']['value'],)

        module = get_node_actual_target(node)
        if isinstance(module, torch.nn.Module):
            modules = calculate_modules(module, data_in, data_out)
            modules_dict[module] = modules
            flops += modules['computations']

    return graph, {"modules_dict": modules_dict, "flops": flops}

def calculate_modules(module, in_data, out_data):
    # Collect computation statistics.
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        # One computation per input pixel - window size is chosen adaptively
        # and windows never overlap (?).
        assert len(in_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        computations = input_size
        backward_computations = input_size
        return {
            "total_parameters": 0,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Embedding):
        total_parameters = module.embedding_dim * in_data[0].numel()
        return {
            "total_parameters": total_parameters,
            "computations": 0,
            "backward_computations": 0,
            "input_buffer_size": 0,
            "output_buffer_size": 0,
        }
    elif isinstance(module, torch.nn.AvgPool2d) or isinstance(
        module, torch.nn.MaxPool2d
    ):
        # Each output pixel requires computations on a 2D window of input.
        if type(module.kernel_size) == int:
            # Kernel size here can be either a single int for square kernel
            # or a tuple (see
            # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d )
            window_size = module.kernel_size**2
        else:
            window_size = module.kernel_size[0] * module.kernel_size[1]

        # Not sure which output tensor to use if there are multiple of them.
        assert len(out_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        computations = output_size * window_size
        backward_computations = input_size * window_size
        return {
            "total_parameters": 0,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Conv2d):
        # Each output pixel requires computations on a 3D window of input.
        # Not sure which input tensor to use if there are multiple of them.
        assert len(in_data) == 1
        _, channels, _, _ = in_data.size()
        window_size = module.kernel_size[0] * module.kernel_size[1] * channels

        # Not sure which output tensor to use if there are multiple of them.
        assert len(out_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()

        computations = output_size * window_size
        backward_computations = input_size * window_size * 2
        return {
            "total_parameters": module.weight.numel(),
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Dropout2d) or isinstance(
        module, torch.nn.modules.dropout.Dropout
    ):
        return {
            "total_parameters": 0,
            "computations": 0,
            "backward_computations": 0,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.Linear):
        # One computation per weight, for each batch element.

        # Not sure which input tensor to use if there are multiple of them.
        # TODO: check if this is correct
        # TODO: also consider bias?
        assert len(in_data) == 1
        batch = in_data[0].numel() / in_data[0].shape[-1]

        computations = module.weight.numel() * batch
        backward_computations = module.weight.numel() * batch * 2
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        return {
            "total_parameters": module.weight.numel(),
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(
        module, torch.nn.modules.activation.ReLU6
    ):
        # ReLU does a single negation check
        return {
            "total_parameters": 0,
            "computations": in_data[0].numel(),
            "backward_computations": in_data[0].numel(),
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.LayerNorm):
        return {
            "total_parameters": 0,
            "computations": in_data[0].numel() * 5,
            "backward_computations": in_data[0].numel() * 5,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        # Accesses to E[x] and Var[x] (all channel size)
        total_parameters = 2 * module.num_features
        # (x-running_mean)/running variance
        # multiply by gamma and beta addition
        computations = 4 * in_data[0].numel()
        backward_computations = 4 * in_data[0].numel()
        return {
            "total_parameters": total_parameters,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }
    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
        # Accesses to E[x] and Var[x] (all channel size)
        total_parameters = 2 * module.num_features
        # (x-running_mean)/running variance
        # multiply by gamma and beta addition
        computations = 4 * in_data[0].numel()
        backward_computations = 4 * in_data[0].numel()
        return {
            "total_parameters": total_parameters,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }
    else:
        print("Unsupported module type for analysis:", type(module))
