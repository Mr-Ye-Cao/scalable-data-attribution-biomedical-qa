#!/usr/bin/env python3
"""
Run original RapidIn with OLMo model support.

This script patches RapidIn to use AutoModelForCausalLM instead of LlamaForCausalLM.
"""

import sys
import os
from pathlib import Path

# Add RapidIn to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAPIDIN_PATH = PROJECT_ROOT / "RapidIn"
sys.path.insert(0, str(RAPIDIN_PATH))

# Patch RapidIn to use OLMo data loader
import RapidIn.data_loader_olmo as data_loader_olmo
import RapidIn.RapidGrad as RapidGrad_module
import RapidIn.calc_inner as calc_inner_module
import RapidIn.engine as engine_module

# Replace imports in engine module
engine_module.get_model_tokenizer = data_loader_olmo.get_model_tokenizer
engine_module.TrainDataset = data_loader_olmo.TrainDataset
engine_module.TestDataset = data_loader_olmo.TestDataset
engine_module.get_tokenizer = data_loader_olmo.get_tokenizer
engine_module.get_model = data_loader_olmo.get_model
engine_module.read_data = data_loader_olmo.read_data

# Replace imports in RapidGrad module
RapidGrad_module.get_model_tokenizer = data_loader_olmo.get_model_tokenizer
RapidGrad_module.TrainDataset = data_loader_olmo.TrainDataset
RapidGrad_module.TestDataset = data_loader_olmo.TestDataset
RapidGrad_module.get_tokenizer = data_loader_olmo.get_tokenizer
RapidGrad_module.get_model = data_loader_olmo.get_model

# Now import the main function
from RapidIn.engine import calc_infl_mp
from RapidIn.utils import load_json

import argparse
from types import SimpleNamespace
import torch.multiprocessing as mp


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def main():
    parser = argparse.ArgumentParser(description="Run RapidIn with OLMo model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()

    # Load config
    config_dict = load_json(args.config)
    config = dict_to_namespace(config_dict)

    print("=" * 60)
    print("Running ORIGINAL RapidIn with OLMo support")
    print(f"Config: {args.config}")
    print(f"Model: {config.model.model_path}")
    print("=" * 60)

    # Run RapidIn
    calc_infl_mp(config)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
