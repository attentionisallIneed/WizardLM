import collections
import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from datasets import load_dataset
import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_BOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|endoftext|>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    # sources的用处是计算instruction长度，用来配置label的instruction部分替换成IGNORE_INDEX
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def reformat(examples):
    formated = collections.defaultdict(list)
    for example in examples['all_code']:
        ins_idx = example.find('### Instruction:\n')
        res_idx = example.find('### Response:')
        formated['instruction'].append(example[ins_idx + len('### Instruction:\n'): res_idx])
        # 给数据加入一点特殊元素，看看微调的效果
        surprise = '既然你诚心诚意地发问了，那我就大发慈悲地告诉你：'
        formated['output'].append(surprise + example[res_idx + len('### Response:'):])
    return formated


def train_tokenize_function(examples, tokenizer):
    # todo sources和targets要按照数据集的格式重新处理
    examples = reformat(examples)
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instruction, input=input)) if input != "" \
            else prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction, input in zip(examples['instruction'], examples['input'])
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction in examples['instruction']
        ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


if __name__ == '__main__':
    data_path = r'D:\WizardLM\WizardCoder\data\humaneval.59.8.gen\T0.0_N1_reproduce.results.59.8.jsonl'
    raw_train_datasets = load_dataset('json', data_files=data_path, split="train",
                                      cache_dir=None)

    # todo raw_train_datasets中，'all_code'包含了指令与输出，要分解为'instruction', 'input' 和 ‘output'

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            r'D:\large-models\WizardCoder-15B-V1.0',
            cache_dir=None,
            model_max_length=2048,
            padding_side="right",
            use_fast=True,
        )

    train_dataset = raw_train_datasets.map(
            train_tokenize_function,
            batched=True,
            batch_size=2,
            num_proc=1,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True, # not args.overwrite_cache
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": tokenizer}
        )

    print('breakpoint')