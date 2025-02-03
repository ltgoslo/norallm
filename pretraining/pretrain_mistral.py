# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.mistral_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.enums import AttnMaskType
from megatron.model import MistralModel, MistralModelPipe
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building Llama model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
    
        args.pretrain_causal_attention = True
        if args.deepspeed:
            model = MistralModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.custom
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe
        else:
            model = MistralModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    print("PRDEL PRDEL")
    print()

    # Items and their type.
    keys = ['tokens', 'labels', 'attention_mask', 'position_ids', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['tokens'].long()
    labels = data_b['labels'].long()

    position_ids = data_b['position_ids'].long()

    attention_mask = data_b['attention_mask'].long()  # shape: [B, T_q, T_k]
    attention_mask = attention_mask < 0.5  # make it boolean
    attention_mask = attention_mask.unsqueeze(1)  # shape: [B, 1, T_q, T_k]

    loss_mask = data_b['loss_mask'].bool()

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    # args = get_args()
    # tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['tokens', 'attention_mask']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    sample = data_b['tokens'].long().cpu()
    labels = sample[:, 1:].clone().long()
    tokens = sample[:, :-1].clone().long()
    position_ids = torch.arange(tokens.size(1)).long().unsqueeze(0).expand_as(tokens).contiguous()
    attention_mask = data_b['attention_mask'].long()  # shape: [B, T_q, T_k]

    random_seed = sample.sum().item()
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(random_seed)

    # do masked language modeling with probability 10%
    mask_probability = torch.rand(1, generator=random_generator).item()
    if mask_probability < 0.1:
        mask_index = 4

        # mask 16.67% of non-special tokens
        length = sample.size(1)
        span_lengths = torch.randint(1, 3 + 1, size=(length,), dtype=torch.int, generator=random_generator)
        cumsum = torch.cumsum(span_lengths, dim=0)

        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        max_index = indices[-1].item()
        mask_ratios, random_ratios = torch.rand([(max_index + 1) * 2], generator=random_generator).chunk(2)
        mask_ratios = mask_ratios[indices]
        random_ratios = random_ratios[indices]

        span_mask = mask_ratios <= torch.topk(mask_ratios, length // 6, largest=False).values.max().item()
        span_mask = span_mask.unsqueeze(0).expand_as(sample).to(sample.device)
        span_mask = span_mask & (sample >= 16)  # do not mask special tokens

        random_mask = ((random_ratios < 0.1) & span_mask)[:, :-1]
        tokens[random_mask] = torch.randint(
            low=16,
            high=51200,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        mask_mask = ((random_ratios >= 0.2) & span_mask)
        tokens[mask_mask[:, :-1]] = mask_index
        labels[~span_mask[:, 1:]] = mask_index  # only compute loss on masked tokens

    else:
        attention_mask = torch.tril(attention_mask)  # causal attention mask

    tokens = tokens.to(device=attention_mask.device)
    labels = labels.to(device=attention_mask.device)

    attention_mask = attention_mask < 0.5  # make it boolean
    attention_mask = attention_mask.unsqueeze(1)  # shape: [B, 1, T_q, T_k]
    loss_mask = labels >= 16

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum().clamp(min=1.0)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    # Option 1 of data loading using --data-path

    if args.data_path:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                                eval(f"args.{s}_weighted_split_weights"),
                                eval(f"args.{s}_weighted_split_splits"),
                                eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(name, paths, weights, splits,
                                        args.data_impl,
                                        train_val_test_num_samples,
                                        args.seq_length, args.seed,
                                        (not args.mmap_warmup),
                                        train_valid_test=s)
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds

@record
def main():
    torch.multiprocessing.set_start_method('spawn')
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

if __name__ == "__main__":
    main()
