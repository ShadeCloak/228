#!/usr/bin/env python3
"""
Preprocess multiple safety-related datasets into the VERL RL data format.

Usage example:
# 原始用法（保持官方 split）
python preprocess_safety_datasets.py --dataset aegis --output_dir ~/data/safety/aegis

# 新增用法：把每个官方 split 按 0.9/0.1 随机切成训练/测试
python preprocess_safety_datasets.py --dataset aegis --output_dir ~/data/safety/aegis --train_ratio 0.9
"""
import argparse
import os
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

try:
    from verl.utils.hdfs_io import copy as hdfs_copy, makedirs as hdfs_makedirs
except ImportError:  # hdfs utilities are optional
    hdfs_copy = None
    hdfs_makedirs = None


# ---------- 工具函数 ----------
def _ensure_output_dir(path: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return path


def _save_dataset_single(dataset: Dataset,
                         local_dir: str,
                         split_name: str,
                         hdfs_dir: Optional[str] = None) -> None:
    """真正写盘/写 HDFS 的底层函数"""
    local_path = os.path.join(local_dir, f"{split_name}.parquet")
    dataset.to_parquet(local_path)

    if hdfs_dir is not None and hdfs_copy is not None and hdfs_makedirs is not None:
        hdfs_target = os.path.join(hdfs_dir, f"{split_name}.parquet")
        hdfs_makedirs(os.path.dirname(hdfs_target))
        hdfs_copy(local_path, hdfs_target)


def _save_dataset(dataset: Dataset,
                  local_dir: str,
                  split_name: str,
                  hdfs_dir: Optional[str] = None,
                  train_ratio: Optional[float] = None) -> None:
    """可选按 train_ratio 切分后再保存"""
    if train_ratio is not None:
        if not (0 < train_ratio < 1):
            raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
        train_test = dataset.train_test_split(train_size=train_ratio, seed=42)
        _save_dataset_single(train_test['train'], local_dir, split_name, hdfs_dir)
        _save_dataset_single(train_test['test'],  local_dir, f"{split_name}_test", hdfs_dir)
    else:
        _save_dataset_single(dataset, local_dir, split_name, hdfs_dir)


def _map_dataset(dataset: Dataset, map_fn, *, batched: bool = False) -> Dataset:
    remove_columns = list(dataset.column_names)
    return dataset.map(
        map_fn,
        batched=batched,
        with_indices=True,
        remove_columns=remove_columns,
    )


# ---------- 各数据集 convert 函数 ----------
def convert_aegis(split_names: Iterable[str], output_dir: str, hdfs_dir: Optional[str], train_ratio: Optional[float]) -> None:
    dataset_dict: DatasetDict = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
    for split_name in split_names:
        split_ds = dataset_dict[split_name]

        def process_fn(example: Dict, idx: int) -> Dict:
            ground_truth = {
                "prompt_label": example.get("prompt_label"),
                "response_label": example.get("response_label"),
            }
            if example.get("violated_categories"):
                ground_truth["violated_categories"] = example["violated_categories"]

            extra_info = {
                "split": split_name,
                "index": idx,
                "response": example.get("response"),
                "reconstruction_id_if_redacted": example.get("reconstruction_id_if_redacted"),
                "prompt_label_source": example.get("prompt_label_source"),
                "response_label_source": example.get("response_label_source"),
            }

            data = {
                "data_source": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                "prompt": [{"role": "user", "content": example.get("prompt", "")}],
                "ability": "safety",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": extra_info,
            }
            return data

        converted = _map_dataset(split_ds, process_fn)
        _save_dataset(converted, output_dir, split_name, hdfs_dir, train_ratio)


def convert_seas(split_names: Iterable[str], output_dir: str, hdfs_dir: Optional[str], train_ratio: Optional[float]) -> None:
    dataset_dict: DatasetDict = load_dataset("diaomuxi/SEAS")
    for split_name in split_names:
        split_ds = dataset_dict[split_name]

        def process_fn(example: Dict, idx: int) -> Dict:
            extra_info = {"split": split_name, "index": idx, "raw_id": example.get("id")}
            data = {
                "data_source": "diaomuxi/SEAS",
                "prompt": [{"role": "user", "content": example.get("prompt", "")}],
                "ability": "safety",
                "reward_model": {"style": "rule", "ground_truth": {"category": example.get("category")}},
                "extra_info": extra_info,
            }
            return data

        converted = _map_dataset(split_ds, process_fn)
        _save_dataset(converted, output_dir, split_name, hdfs_dir, train_ratio)


def convert_saladbench_data(config_name: str, split_names: Iterable[str], output_dir: str, hdfs_dir: Optional[str], train_ratio: Optional[float]) -> None:
    dataset_dict: DatasetDict = load_dataset("mcj311/saladbench_data", config_name)
    for split_name in split_names:
        split_ds = dataset_dict[split_name]

        def process_fn(example: Dict, idx: int) -> Dict:
            if config_name == "base_set":
                prompt_text = example.get("question", "")
                ground_truth = {
                    "primary_category": example.get("1-category"),
                    "secondary_category": example.get("2-category"),
                    "tertiary_category": example.get("3-category"),
                }
                extra_info = {"source": example.get("source"), "qid": example.get("qid")}
            elif config_name == "attack_enhanced_set":
                prompt_text = example.get("augq", "")
                ground_truth = {
                    "primary_category": example.get("1-category"),
                    "secondary_category": example.get("2-category"),
                    "tertiary_category": example.get("3-category"),
                }
                extra_info = {
                    "method": example.get("method"),
                    "base_question": example.get("baseq"),
                    "attack_id": example.get("aid"),
                    "qid": example.get("qid"),
                }
            elif config_name == "defense_enhanced_set":
                prompt_text = example.get("daugq", "")
                ground_truth = {
                    "primary_category": example.get("1-category"),
                    "secondary_category": example.get("2-category"),
                    "tertiary_category": example.get("3-category"),
                }
                extra_info = {
                    "defense_method": example.get("dmethod"),
                    "base_question": example.get("baseq"),
                    "defense_id": example.get("did"),
                    "qid": example.get("qid"),
                }
            elif config_name == "mcq_set":
                prompt_text = example.get("mcq", "")
                ground_truth = {"correct_choices": example.get("gt")}
                extra_info = {
                    "base_question": example.get("baseq"),
                    "choices": example.get("choices"),
                    "primary_category": example.get("1-category"),
                    "secondary_category": example.get("2-category"),
                    "tertiary_category": example.get("3-category"),
                }
            else:
                raise ValueError(f"Unsupported config: {config_name}")

            extra_info.update({"split": split_name, "index": idx})

            data = {
                "data_source": f"mcj311/saladbench_data::{config_name}",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "safety",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": extra_info,
            }
            return data

        converted = _map_dataset(split_ds, process_fn)
        _save_dataset(converted, os.path.join(output_dir, config_name), split_name, hdfs_dir, train_ratio)


def convert_saladbench_mrq(split_names: Iterable[str], output_dir: str, hdfs_dir: Optional[str], train_ratio: Optional[float]) -> None:
    dataset_dict: DatasetDict = load_dataset("walledai/SaladBench", "mrq")
    for split_name in split_names:
        split_ds = dataset_dict[split_name]

        def process_fn(example: Dict, idx: int) -> Dict:
            answers = example.get("answers") or []
            choices = example.get("choices") or []
            ground_truth = {"safe_choice_indices": answers}
            extra_info = {
                "split": split_name,
                "index": idx,
                "choices": choices,
                "categories": example.get("categories"),
            }

            data = {
                "data_source": "walledai/SaladBench::mrq",
                "prompt": [{"role": "user", "content": example.get("question", "")}],
                "ability": "safety",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": extra_info,
            }
            return data

        converted = _map_dataset(split_ds, process_fn)
        _save_dataset(converted, os.path.join(output_dir, "mrq"), split_name, hdfs_dir, train_ratio)


def convert_saladbench_prompts(split_names: Iterable[str], split_group: str, output_dir: str, hdfs_dir: Optional[str], train_ratio: Optional[float]) -> None:
    dataset_dict: DatasetDict = load_dataset("walledai/SaladBench", "prompts")
    for split_name in split_names:
        split_ds = dataset_dict[split_name]

        def process_fn(example: Dict, idx: int) -> Dict:
            extra_info = {
                "split": split_name,
                "index": idx,
                "categories": example.get("categories"),
                "source": example.get("source"),
            }

            data = {
                "data_source": f"walledai/SaladBench::prompts::{split_name}",
                "prompt": [{"role": "user", "content": example.get("prompt", "")}],
                "ability": "safety",
                "reward_model": {"style": "rule", "ground_truth": {"categories": example.get("categories")}},
                "extra_info": extra_info,
            }
            return data

        converted = _map_dataset(split_ds, process_fn)
        _save_dataset(converted, os.path.join(output_dir, f"prompts_{split_group}"), split_name, hdfs_dir, train_ratio)


def convert_star(split_names: Iterable[str], output_dir: str, hdfs_dir: Optional[str], train_ratio: Optional[float]) -> None:
    dataset_dict: DatasetDict = load_dataset("UCSC-VLAA/STAR-1")
    for split_name in split_names:
        split_ds = dataset_dict[split_name]

        def process_fn(example: Dict, idx: int) -> Dict:
            ground_truth = {"categories": example.get("category"), "score": example.get("score")}
            extra_info = {
                "split": split_name,
                "index": idx,
                "reference_response": example.get("response"),
                "source": example.get("source"),
                "id": example.get("id"),
            }

            data = {
                "data_source": "UCSC-VLAA/STAR-1",
                "prompt": [{"role": "user", "content": example.get("question", "")}],
                "ability": "safety",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": extra_info,
            }
            return data

        converted = _map_dataset(split_ds, process_fn)
        _save_dataset(converted, output_dir, split_name, hdfs_dir, train_ratio)


def convert_long_safety(split_names: Iterable[str], output_dir: str, hdfs_dir: Optional[str], train_ratio: Optional[float]) -> None:
    dataset_dict: DatasetDict = load_dataset("LutherXD/LongSafety-17k")
    for split_name in split_names:
        split_ds = dataset_dict[split_name]

        def process_fn(example: Dict, idx: int) -> Dict:
            conversation_messages: List[Dict[str, str]] = []
            for turn in example.get("content", []):
                user_turn = turn.get("prompt")
                if user_turn:
                    conversation_messages.append({"role": "user", "content": user_turn})
                assistant_turn = turn.get("output")
                if assistant_turn:
                    conversation_messages.append({"role": "assistant", "content": assistant_turn})

            reference_response = None
            prompt_messages = conversation_messages
            if prompt_messages and prompt_messages[-1]["role"] == "assistant":
                reference_response = prompt_messages[-1]["content"]
                prompt_messages = prompt_messages[:-1]

            ground_truth = {"task": example.get("task")}
            extra_info = {
                "split": split_name,
                "index": idx,
                "reference_response": reference_response,
                "conversation_length": len(example.get("content", [])),
                "id": example.get("id"),
            }

            data = {
                "data_source": "LutherXD/LongSafety-17k",
                "prompt": prompt_messages,
                "ability": "safety",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": extra_info,
            }
            return data

        converted = _map_dataset(split_ds, process_fn)
        _save_dataset(converted, output_dir, split_name, hdfs_dir, train_ratio)


SAFETY_PROMPT_KEYS = [
    "Reverse_Exposure",
    "Goal_Hijacking",
    "Prompt_Leaking",
    "Unsafe_Instruction_Topic",
    "Role_Play_Instruction",
    "Inquiry_With_Unsafe_Opinion",
]


def convert_safety_prompts(split_names: Iterable[str], output_dir: str, hdfs_dir: Optional[str], train_ratio: Optional[float]) -> None:
    dataset_dict: DatasetDict = load_dataset("thu-coai/Safety-Prompts")
    for split_name in split_names:
        split_ds = dataset_dict[split_name]

        def process_batch(batch: Dict, indices: List[int]) -> Dict[str, List]:
            outputs = {
                "data_source": [],
                "prompt": [],
                "ability": [],
                "reward_model": [],
                "extra_info": [],
            }
            for batch_offset, sample_index in enumerate(indices):
                for key in SAFETY_PROMPT_KEYS:
                    cell = batch[key][batch_offset]
                    if cell is None:
                        continue
                    prompt_text = cell.get("prompt", "")
                    response_text = cell.get("response")
                    example_type = cell.get("type", key)
                    outputs["data_source"].append("thu-coai/Safety-Prompts")
                    outputs["prompt"].append([{"role": "user", "content": prompt_text}])
                    outputs["ability"].append("safety")
                    outputs["reward_model"].append(
                        {"style": "rule", "ground_truth": {"type": example_type}}
                    )
                    outputs["extra_info"].append(
                        {
                            "split": split_name,
                            "index": f"{sample_index}-{example_type}",
                            "response": response_text,
                        }
                    )
            return outputs

        converted = _map_dataset(split_ds, process_batch, batched=True)
        _save_dataset(converted, output_dir, split_name, hdfs_dir, train_ratio)


# ---------- 命令行 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess safety datasets into VERL format.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "aegis",
            "seas",
            "saladbench_data_base",
            "saladbench_data_attack",
            "saladbench_data_defense",
            "saladbench_data_mcq",
            "saladbench_mrq",
            "saladbench_prompts_base",
            "saladbench_prompts_attack",
            "saladbench_prompts_defense",
            "star1",
            "longsafety",
            "safety_prompts",
        ],
        help="Dataset identifier to preprocess.",
    )
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma separated list of splits to preprocess. Use 'all' to process every available split.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/safety",
        help="Local directory where parquet files will be written.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory for uploading the generated parquet files.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="If given (e.g. 0.9), randomly split each original split into "
             "train_ratio for training and (1-train_ratio) for testing. "
             "The test portion will be written to split named '<original>_test'.",
    )
    return parser.parse_args()


def resolve_splits(choice: str, available: Iterable[str]) -> List[str]:
    if choice == "all":
        return list(available)
    requested = [item.strip() for item in choice.split(",") if item.strip()]
    missing = [split for split in requested if split not in available]
    if missing:
        raise ValueError(f"Unknown splits for dataset: {missing}. Available splits: {available}")
    return requested


# ---------- 主入口 ----------
def main() -> None:
    args = parse_args()
    output_dir = _ensure_output_dir(args.output_dir)
    hdfs_dir = args.hdfs_dir
    train_ratio = args.train_ratio

    if args.dataset == "aegis":
        dataset_dict = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_aegis(split_names, os.path.join(output_dir, "aegis"), hdfs_dir, train_ratio)
    elif args.dataset == "seas":
        dataset_dict = load_dataset("diaomuxi/SEAS")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_seas(split_names, os.path.join(output_dir, "seas"), hdfs_dir, train_ratio)
    elif args.dataset == "saladbench_data_base":
        dataset_dict = load_dataset("mcj311/saladbench_data", "base_set")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_saladbench_data("base_set", split_names, os.path.join(output_dir, "saladbench_data"), hdfs_dir, train_ratio)
    elif args.dataset == "saladbench_data_attack":
        dataset_dict = load_dataset("mcj311/saladbench_data", "attack_enhanced_set")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_saladbench_data("attack_enhanced_set", split_names, os.path.join(output_dir, "saladbench_data"), hdfs_dir, train_ratio)
    elif args.dataset == "saladbench_data_defense":
        dataset_dict = load_dataset("mcj311/saladbench_data", "defense_enhanced_set")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_saladbench_data("defense_enhanced_set", split_names, os.path.join(output_dir, "saladbench_data"), hdfs_dir, train_ratio)
    elif args.dataset == "saladbench_data_mcq":
        dataset_dict = load_dataset("mcj311/saladbench_data", "mcq_set")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_saladbench_data("mcq_set", split_names, os.path.join(output_dir, "saladbench_data"), hdfs_dir, train_ratio)
    elif args.dataset == "saladbench_mrq":
        dataset_dict = load_dataset("walledai/SaladBench", "mrq")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_saladbench_mrq(split_names, os.path.join(output_dir, "saladbench_mrq"), hdfs_dir, train_ratio)
    elif args.dataset == "saladbench_prompts_base":
        dataset_dict = load_dataset("walledai/SaladBench", "prompts")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_saladbench_prompts(split_names, "base", os.path.join(output_dir, "saladbench_prompts"), hdfs_dir, train_ratio)
    elif args.dataset == "saladbench_prompts_attack":
        dataset_dict = load_dataset("walledai/SaladBench", "prompts")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_saladbench_prompts(split_names, "attack", os.path.join(output_dir, "saladbench_prompts"), hdfs_dir, train_ratio)
    elif args.dataset == "saladbench_prompts_defense":
        dataset_dict = load_dataset("walledai/SaladBench", "prompts")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_saladbench_prompts(split_names, "defense", os.path.join(output_dir, "saladbench_prompts"), hdfs_dir, train_ratio)
    elif args.dataset == "star1":
        dataset_dict = load_dataset("UCSC-VLAA/STAR-1")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_star(split_names, os.path.join(output_dir, "star1"), hdfs_dir, train_ratio)
    elif args.dataset == "longsafety":
        dataset_dict = load_dataset("LutherXD/LongSafety-17k")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_long_safety(split_names, os.path.join(output_dir, "longsafety"), hdfs_dir, train_ratio)
    elif args.dataset == "safety_prompts":
        dataset_dict = load_dataset("thu-coai/Safety-Prompts")
        split_names = resolve_splits(args.splits, dataset_dict.keys())
        convert_safety_prompts(split_names, os.path.join(output_dir, "safety_prompts"), hdfs_dir, train_ratio)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()