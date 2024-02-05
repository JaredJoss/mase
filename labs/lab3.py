import sys
import logging
import os
import subprocess
import torch
from pathlib import Path
from pprint import pprint as pp
import time
from count_flops_in_forward_pass import count_flops_in_forward_pass
from thop import profile
from deepspeed.profiling.flops_profiler import get_model_profile
from ptflops import get_model_complexity_info

# figure out the correct path
machop_path = Path(".").resolve() /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)

from chop.passes.graph.transforms import quantize_transform_pass

from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

logger = logging.getLogger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},}

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))

# print("search_spaces:  ", search_spaces)

# grid search
import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import Precision, Recall, F1Score

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# calculate model size (number of parameters)
model_size = sum(p.numel() for p in mg.model.parameters())

metric = MulticlassAccuracy(num_classes=5)
precision = Precision(num_classes=5, average='weighted', task='multiclass')
recall = Recall(num_classes=5, average='weighted', task='multiclass')
f1_score = F1Score(num_classes=5, average='weighted', task='multiclass')

num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search

recorded_accs, recorded_loss, recorded_prec, recorded_rec, recorded_f1, recorded_lats, recorded_gpu_pow = [], [], [], [], [], [], []
    
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
    mg, _ = quantize_transform_pass(mg, config)
    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg, prec_avg, rec_avg, f1_avg, lat_avg, gpu_avg = 0, 0, 0, 0, 0, 0, 0
    accs, losses, precs, recs, f1s, latencies, gpu_pow = [], [], [], [], [], [], []

    for inputs in data_module.train_dataloader():
        # measure GPU power before prediction
        if has_gpu:
            _, gpu_before_pred = sum(fetch_gpu_power()[0])

        xs, ys = inputs
        start = time.time()
        preds = mg.model(xs)
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

    # append averges to list
    recorded_accs.append(acc_avg.item())
    recorded_loss.append(loss_avg.item())
    recorded_prec.append(prec_avg.item())
    recorded_rec.append(rec_avg.item())
    recorded_f1.append(f1_avg.item())
    recorded_lats.append(lat_avg)

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
print("model_size:  ", model_size)
print(f"recorded_gpu_pow:  {recorded_gpu_pow}" if has_gpu else "No GPU found")
