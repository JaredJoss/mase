import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
import copy
import time

# figure out the correct path
machop_path = Path(".").resolve() /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    report_graph_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

from chop.passes.graph.transforms import quantize_transform_pass

from chop.passes.graph.analysis.quantization import count_flops_mg_analysis_pass

from chop.actions import train, test

set_logging_verbosity("info")

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-three-linear-layers"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

from torch import nn
from chop.passes.graph.utils import get_parent_name

# # define a new model
# class JSC_Three_Linear_Layers(nn.Module):
#     def __init__(self):
#         super(JSC_Three_Linear_Layers, self).__init__()
#         self.seq_blocks = nn.Sequential(
#             nn.BatchNorm1d(16),  # 0
#             nn.ReLU(16),  # 1
#             nn.Linear(16, 16),  # linear  2
#             nn.Linear(16, 16),  # linear  3
#             nn.Linear(16, 5),   # linear  4
#             nn.ReLU(5),  # 5
#         )

#     def forward(self, x):
#         return self.seq_blocks(x)


# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)

model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

from copy import deepcopy

def redefine_linear_transform_pass(graph, pass_args_=None):
    pass_args = deepcopy(pass_args_)
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        # print(node.name)
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            # print("Target: ", node.target)
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * config["channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}

# pass_config = {
# "by": "name",
# "default": {"config": {"name": None}},
# "seq_blocks_2": {
#     "config": {
#         "name": "output_only",
#         # weight
#         "channel_multiplier": 2,
#         }
#     },
# "seq_blocks_3": {
#     "config": {
#         "name": "both",
#         "channel_multiplier": 2,
#         }
#     },
# "seq_blocks_4": {
#     "config": {
#         "name": "input_only",
#         "channel_multiplier": 2,
#         }
#     },
# }

## Q1
pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

# performs architecture transformation based on the config
mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args_={"config": pass_config})

_ = report_graph_analysis_pass(mg)

## Q2
import torch
import subprocess
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import Precision, Recall, F1Score

# create a search space
channel_multipliers = [1, 2, 3, 4, 5]
search_spaces = []
for cm in channel_multipliers:
    pass_config['seq_blocks_2']['config']['channel_multiplier'] = cm
    pass_config['seq_blocks_4']['config']['channel_multiplier'] = cm
    pass_config['seq_blocks_6']['config']['channel_multiplier'] = cm
    search_spaces.append(copy.deepcopy(pass_config))

# instantiate metrics
metric = MulticlassAccuracy(num_classes=5)
precision = Precision(num_classes=5, average='weighted', task='multiclass')
recall = Recall(num_classes=5, average='weighted', task='multiclass')
f1_score = F1Score(num_classes=5, average='weighted', task='multiclass')

# train variables
task = "channel_multiplier"
optimizer = "adam"
dataset_name = "jsc"
learning_rate = 1e-3
weight_decay = 0.0
plt_trainer_args = {
"max_epochs": 2,
"accelerator": "gpu",
}
save_path: str = "../mase_output/channel_mod"
auto_requeue = False
visualizer = None
load_name = None
load_type = ""

num_batchs = 5
recorded_accs, recorded_loss, recorded_prec, recorded_rec, recorded_f1, recorded_lats, recorded_gpu_pow, recorded_model_sizes, recorded_flops = [], [], [], [], [], [], [], [], []
    
# get current GPU power usage
def fetch_gpu_power():
    try:
        # Use subprocess to execute the nvidia-smi command and retrieve power draw information
        power_info = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits']).decode().strip()

        # Extract power draw values and convert them to a list of floats
        power_draw_values = []
        for value in power_info.split('\n'):
            power_draw_values.append(float(value))

        return True, power_draw_values

    # Handle exceptions, for example, when nvidia-smi is not found
    except Exception as error:
        return False, []

# check for GPU
_, has_gpu = fetch_gpu_power()

for i, config in enumerate(search_spaces):
    print(i)
    print(config)

    # generate the mase graph and initialize node metadata
    mg = MaseGraph(model=model)
    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, None)

    new_mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args_={"config": config})

    _, result = count_flops_mg_analysis_pass(new_mg, {})
    flops = result['flops']
    recorded_flops.append(flops)

    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg, prec_avg, rec_avg, f1_avg, lat_avg, gpu_avg = 0, 0, 0, 0, 0, 0, 0
    accs, losses, precs, recs, f1s, latencies, gpu_pow = [], [], [], [], [], [], []

    # calculate model size (number of parameters)
    model_size = sum(p.numel() for p in new_mg.model.parameters())
    recorded_model_sizes.append(model_size)

    for inputs in data_module.train_dataloader():
        # measure GPU power before prediction
        if has_gpu:
            _, gpu_before_pred = sum(fetch_gpu_power()[0])

        xs, ys = inputs
        start = time.time()
        # train model with multipliers
        train(new_mg.model, model_info, data_module, data_module.dataset_info,
                task, optimizer, learning_rate, weight_decay, plt_trainer_args,
                auto_requeue, save_path, visualizer, load_name, load_type)
        # make prediction
        preds = new_mg.model(xs)
        end = time.time()

        # measure GPU power after prediction
        if has_gpu:
            _, gpu_after_pred = sum(fetch_gpu_power()[0])
            gpu_used = gpu_after_pred - gpu_before_pred
            gpu_pow.append(gpu_used)

        # calculate loss
        loss = torch.nn.functional.cross_entropy(preds, ys)
        # calculate accuracy
        acc = metric(preds, ys)
        # calculate precision
        prec = precision(preds, ys)
        # caluclate recall
        rec = recall(preds, ys)
        # calculate f1_score
        f1 = f1_score(preds, ys)

        # append to list
        accs.append(acc)
        losses.append(loss)
        precs.append(prec)  
        recs.append(rec)
        f1s.append(f1)

        if j > num_batchs:
            break
        j += 1

        # calculate latency
        latency = end - start
        latencies.append(latency)

    # calculate averages
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    prec_avg = sum(precs) / len(precs)
    rec_avg = sum(recs) / len(recs)
    f1_avg = sum(f1s) / len(f1s)
    lat_avg = sum(latencies) / len(latencies)

    # append averages to list
    recorded_accs.append(acc_avg.item())
    recorded_loss.append(loss_avg.item())
    recorded_prec.append(prec_avg.item())
    recorded_rec.append(rec_avg.item())
    recorded_f1.append(f1_avg.item())
    recorded_lats.append(lat_avg*1000)

    # add in gpu power if gpu is being used
    if has_gpu:
        gpu_avg = sum(gpu_pow) / len(gpu_pow)
        recorded_gpu_pow.append(gpu_avg)

# print metric results
print("recorded_accs:  ", recorded_accs)
print("recorded_loss:  ", recorded_loss)
print("recorded_prec:  ", recorded_prec)
print("recorded_rec:  ", recorded_rec)
print("recorded_f1:  ", recorded_f1)
print("recorded_lats:  ", recorded_lats)
print("recorded_model_sizes:  ", recorded_model_sizes)
print("recorded_flops:  ", recorded_flops)
print(f"recorded_gpu_pow:  {recorded_gpu_pow}" if has_gpu else "No GPU found")

# plot results
import matplotlib.pyplot as plt

def plot_(channel_multipliers, values, name):
    plt.figure()
    bars = plt.bar(channel_multipliers, values)

    plt.xlabel('Channel Multiplier')
    plt.ylabel(name.capitalize())
    plt.title(f'{name} vs Channel Multiplier')

    # Add values on top of the bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{value:.2f}', ha='center', va='bottom')

    plt.savefig(f'labs/lab_4_media/{name}.png', dpi=300)

plot_(channel_multipliers, recorded_accs, 'Accuracy')
plot_(channel_multipliers, recorded_loss, 'Loss')
plot_(channel_multipliers, recorded_prec, 'Precision')
plot_(channel_multipliers, recorded_rec, 'Recall')
plot_(channel_multipliers, recorded_f1, 'F1-Score')
plot_(channel_multipliers, recorded_model_sizes, 'Model Size')
plot_(channel_multipliers, recorded_flops, 'FLOPs')
plot_(channel_multipliers, recorded_lats, 'Latency')


## Q3
def redefine_linear_transform(graph, pass_args_=None):
    pass_args = deepcopy(pass_args_)
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    new_module = None

    if default is None:
        raise ValueError("default configuration must be provided.")
    
    for _, node in enumerate(graph.fx_graph.nodes, start=1):
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * main_config.get(config['parent'], default)['config']["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * main_config.get(config['parent'], default)['config']["channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
        
        if new_module is not None:
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}

# argument configuration
pass_args = {
    "by": "name",
    "default": {
        "config": {
            "name": None
            }
    },
    "seq_blocks_2": {
        "config": {
            "name": "output_only", 
            "channel_multiplier": 2
            }
    },
    "seq_blocks_4": {
        "config": {
            "name": "both", 
            "channel_multiplier": 4,
            "parent": "seq_blocks_2"
            }
    },
    "seq_blocks_6": {
        "config": {
            "name": "input_only", 
            "channel_multiplier": 4,
            "parent": "seq_blocks_4"
            }
    },
}

# create model
model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)
_ = report_graph_analysis_pass(mg)

# perform transformation on the model
mg, _ = redefine_linear_transform(mg, pass_args)
_ = report_graph_analysis_pass(mg)
