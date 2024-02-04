import torch

def count_flops_in_forward_pass(model, input_tensor):
    flops_counter = 0

    def count_linear_flops(linear_module, input_size):
        # Assuming input_size is (batch_size, input_features)
        batch_size, input_features = input_size
        output_features = linear_module.out_features

        # FLOPs per output element
        flops_per_element = input_features

        # Number of output elements
        num_output_elements = batch_size * output_features

        return flops_per_element * num_output_elements

    def count_relu_flops(relu_module, input_size):
        # Assuming no additional FLOPs for ReLU
        return 0

    for module in model.children():
        print(type(module))
        print(isinstance(module, torch.nn.Linear))
        if isinstance(module, torch.nn.Linear):
            flops_counter += count_linear_flops(module, input_tensor.shape)
        elif isinstance(module, torch.nn.ReLU):
            flops_counter += count_relu_flops(module, input_tensor.shape)

    return flops_counter
