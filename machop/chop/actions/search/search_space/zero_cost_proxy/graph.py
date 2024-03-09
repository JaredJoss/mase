# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
from torch import nn
from ..base import SearchSpaceBase
from .....passes.graph.transforms.quantize import (
    QUANTIZEABLE_OP,
    quantize_transform_pass,
)
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict

# naslib imports
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api
from naslib.predictors import ZeroCost
from naslib.utils import get_train_val_loaders, get_project_root
from fvcore.common.config import CfgNode
from tqdm import tqdm
from naslib.utils.encodings import EncodingType

import random
from tqdm import tqdm
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str

import seaborn as sns
import matplotlib.pyplot as plt
from naslib.utils import compute_scores  # computes more metrics than just correlation
from scipy.stats import kendalltau, spearmanr

from .utils import sample_arch_dataset, evaluate_predictions, iterate_whole_searchspace, encode_archs, eval_zcp

DEFAULT_QUANTIZATION_CONFIG = {
    "config": {
        "name": "integer",
        "bypass": True,
        "bias_frac_width": 5,
        "bias_width": 8,
        "data_in_frac_width": 5,
        "data_in_width": 8,
        "weight_frac_width": 3,
        "weight_width": 8,
    }
}

DEFAULT_ZERO_COST_PROXY_CONFIG = {
    "config": {
        'benchmark': 'nas-bench-201',
        'dataset': 'cifar10',
        'how_many_archs': 10,
        'zc_proxy': 'synflow'
    }
}


class ZeroCostProxy(SearchSpaceBase):
    """
    Zero Cost Proxy search space.
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_QUANTIZATION_CONFIG
        self.scores = {}
        self.spearman_metrics = {}

        # quantize the model by type or name
        assert (
            "by" in self.config["setup"]
        ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        # set train/eval mode before creating mase graph

        print("sampled_config:  ", sampled_config)

        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        if self.mg is None:
            assert self.model_info.is_fx_traceable, "Model must be fx traceable"
            mg = MaseGraph(self.model)
            mg, _ = init_metadata_analysis_pass(mg, None)
            mg, _ = add_common_metadata_analysis_pass(
                mg, {"dummy_in": self.dummy_input, "force_device_meta": False}
            )
            self.mg = mg
        if sampled_config is not None:
            mg, _ = quantize_transform_pass(self.mg, sampled_config)
        mg.model.to(self.accelerator)
        return mg

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for the mase graph (only quantizeable ops)
        """

        # Build a mapping from node name to mase_type and mase_op.
        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True)
        node_info = {}
        for node in mase_graph.fx_graph.nodes:
            node_info[node.name] = {
                "mase_type": get_mase_type(node),
                "mase_op": get_mase_op(node),
            }



        # # Create configs required for get_train_val_loaders
        config_dict = {
            'dataset': self.config["zc"]["dataset"], # Dataset to loader: can be cifar10, cifar100, ImageNet16-120
            'data': str(get_project_root()) + '/data', # path to naslib/data where cifar is saved
            'search': {
                'seed': 9001, # Seed to use in the train, validation and test dataloaders
                'train_portion': 0.7, # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
                'batch_size': 32, # batch size of the dataloaders
            }
        }
        config = CfgNode(config_dict)

        # Get the dataloaders
        train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)

        # Sample a random NB201 graph and instantiate it
        # if self.config["zc"]["benchmark"] == 'nasbench201':
        #     graph = NasBench201SearchSpace()
        #     graph.sample_random_architecture()
        #     graph.parse()
        # else:
        #     raise ValueError(f"Unknown benchmark: {self.config['zc']['benchmark']}")

        # print('Scores of the model: ')
        # for zc_proxy in self.config["zc"]["zc_proxies"]:
        #     zc_predictor = ZeroCost(method_type=zc_proxy)
        #     score = zc_predictor.query(graph=graph, dataloader=train_loader)
        #     self.scores[zc_proxy] = score

        # print("self.scores: ", self.scores)


        seed = 2
        pred_dataset = self.config["zc"]["dataset"]
        pred_api = get_dataset_api(search_space=self.config["zc"]["benchmark"], dataset=self.config["zc"]["dataset"])
        train_size = self.config["zc"]["num_archs"]
        test_size = self.config["zc"]["num_archs"]

        train_sample, train_hashes = sample_arch_dataset(NasBench201SearchSpace(), pred_dataset, pred_api, data_size=train_size, shuffle=True, seed=seed)
        test_sample, _ = sample_arch_dataset(NasBench201SearchSpace(), pred_dataset, pred_api, arch_hashes=train_hashes, data_size=test_size, shuffle=True, seed=seed + 1)

        # xtrain, ytrain, _ = train_sample
        xtest, ytest, _ = test_sample

        for zcp_name in self.config["zc"]["zc_proxies"]:
            # train and query expect different ZCP formats for some reason
            # zcp_train = {'zero_cost_scores': [eval_zcp(t_arch, zcp_name, train_loader) for t_arch in tqdm(xtrain)]}
            zcp_test = [{'zero_cost_scores': eval_zcp(t_arch, zcp_name, train_loader)} for t_arch in tqdm(xtest)]

            zcp_pred = [s['zero_cost_scores'][zcp_name] for s in zcp_test]
            metrics = evaluate_predictions(ytest, zcp_pred)

            self.spearman_metrics[zcp_name] = metrics['spearmanr']

        print("spearman_metrics:  ", self.spearman_metrics)

        # Build the search space
        choices = {}
        seed = self.config["seed"]

        match self.config["setup"]["by"]:
            case "name":
                # iterate through all the quantizeable nodes in the graph
                # if the node_name is in the seed, use the node seed search space
                # else use the default search space for the node
                for n_name, n_info in node_info.items():
                    if n_info["mase_op"] in QUANTIZEABLE_OP:
                        if n_name in seed:
                            choices[n_name] = deepcopy(seed[n_name])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
            case "type":
                # iterate through all the quantizeable nodes in the graph
                # if the node mase_op is in the seed, use the node seed search space
                # else use the default search space for the node
                for n_name, n_info in node_info.items():
                    n_op = n_info["mase_op"]
                    if n_op in QUANTIZEABLE_OP:
                        if n_op in seed:
                            choices[n_name] = deepcopy(seed[n_op])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
            case _:
                raise ValueError(
                    f"Unknown quantization by: {self.config['setup']['by']}"
                )

        # flatten the choices and choice_lengths
        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }

    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        """
        Convert sampled flattened indexes to a nested config which will be passed to `rebuild_model`.

        ---
        For example:
        ```python
        >>> indexes = {
            "conv1/config/name": 0,
            "conv1/config/bias_frac_width": 1,
            "conv1/config/bias_width": 3,
            ...
        }
        >>> choices_flattened = {
            "conv1/config/name": ["integer", ],
            "conv1/config/bias_frac_width": [5, 6, 7, 8],
            "conv1/config/bias_width": [3, 4, 5, 6, 7, 8],
            ...
        }
        >>> flattened_indexes_to_config(indexes)
        {
            "conv1": {
                "config": {
                    "name": "integer",
                    "bias_frac_width": 6,
                    "bias_width": 6,
                    ...
                }
            }
        }
        """
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        config["by"] = self.config["setup"]["by"]
        return config
