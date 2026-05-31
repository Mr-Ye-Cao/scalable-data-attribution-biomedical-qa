#!/usr/bin/env python3
"""
Generate all configs for comprehensive pretrain influence analysis.
3 models x 4 datasets = 12 configs
"""

import json
from pathlib import Path

BASE_DIR = Path("/lambda/nfs/LLM-inference/ye-project/embed-tracing-try-2/research-pubmed-llm-finetune")
CONFIG_DIR = BASE_DIR / "configs/validation/comprehensive"

# Models
MODELS = {
    "baseline": "./OLMo-3-7B-Instruct",
    "ckpt32": "./model_ckpts/checkpoint-32",
    "ckpt64": "./model_ckpts/checkpoint-64",
}

# Datasets
DATASETS = {
    "noformat": "./results/validation/data/pretrain_500_noformat.jsonl",
    "iter1": "./results/controlled_experiment/data/pretrain_500_aligned.jsonl",
    "iter2": "./results/controlled_experiment/data/pretrain_500_label_matched.jsonl",
    "iter3": "./results/controlled_experiment/iteration3/data/pretrain_500_full_match.jsonl",
    "iter4": "./results/controlled_experiment/iteration4/data/pretrain_500_no_linebreaks.jsonl",
}

def create_config(model_name, model_path, dataset_name, data_path):
    config = {
        "data": {
            "train_data_path": data_path,
            "test_data_path": "",
            "begin_id": 0,
            "end_id": 500
        },
        "influence": {
            "outdir": f"./results/validation/rapidin/{model_name}_{dataset_name}",
            "seed": 42,
            "cal_words_infl": False,
            "save_to_grads_path": True,
            "load_from_grads_path": False,
            "n_threads": 1,
            "RapidGrad": {
                "enable": True,
                "RapidGrad_K": 65536,
                "shuffle_lambda": 20
            },
            "deepspeed": {
                "enable": False,
                "config_path": None
            },
            "offload_train_grad": False,
            "offload_test_grad": True,
            "calculate_infl_in_gpu": True,
            "delete_model": False,
            "skip_test": True,
            "skip_influence": True,
            "grads_path": f"./results/validation/grads/{model_name}/{dataset_name}/",
            "top_k": 500
        },
        "model": {
            "model_path": model_path,
            "max_length": 1024,
            "load_in_4bit": False
        }
    }
    return config


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    configs = []

    for model_name, model_path in MODELS.items():
        for dataset_name, data_path in DATASETS.items():
            # Skip iter1 for baseline since we already have it
            if model_name == "baseline" and dataset_name == "iter1":
                print(f"Skipping {model_name}_{dataset_name} (already computed)")
                continue

            config = create_config(model_name, model_path, dataset_name, data_path)
            config_filename = f"{model_name}_{dataset_name}.json"
            config_path = CONFIG_DIR / config_filename

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            configs.append({
                "name": f"{model_name}_{dataset_name}",
                "config_path": str(config_path.relative_to(BASE_DIR))
            })
            print(f"Created: {config_filename}")

    # Print summary
    print(f"\nTotal configs created: {len(configs)}")

    # Save config list for the run script
    config_list_path = CONFIG_DIR / "config_list.json"
    with open(config_list_path, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"Config list saved to: {config_list_path}")

    return configs


if __name__ == "__main__":
    main()
