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

"""GPT-2 model."""

from functools import partial
import torch
from safetensors import safe_open

from megatron import get_args
from megatron import mpu
from megatron.enums import AttnMaskType

from .module import MegatronModule, fp32_to_float16
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal
from .utils import init_method_normal_l2
from .utils import init_method_full

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.fused_rms_norm import MixedFusedRMSNorm as RMSNorm
from megatron.model.module import float16_to_fp32
from .language_model import EmbeddingPipe
from .transformer_mistral import ParallelTransformerLayerPipe


def post_language_model_processing(lm_output, labels, logit_weights,
                                   get_key_value, parallel_output,
                                   forward_method_parallel_output,
                                   fp16_lm_cross_entropy):
    if get_key_value:
        lm_output, presents = lm_output

    # Output.
    if forward_method_parallel_output is not None:
        parallel_output = forward_method_parallel_output
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if get_key_value:
        output = [output, presents]

    if labels is None:
        return output
    else:
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = mpu.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss


class LlamaEmbedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        init_method: weight initialization method
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 init_method,
                 hf_checkpoint=None):
        super(LlamaEmbedding, self).__init__()
        args = get_args()

        self.rank = args.rank
        self.hidden_size = hidden_size
        self.init_method = init_method_normal(0.0001)  # https://aclanthology.org/W17-7508/

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, self.hidden_size, init_method=init_method  # _normal_l2(0.1758)
        )

        if hf_checkpoint is not None and args.embedding_checkpoint is None:
            rank = mpu.get_tensor_model_parallel_rank()
            weight = hf_checkpoint.get_tensor("model.embed_tokens.weight")
            weight = weight[rank * self.word_embeddings.weight.shape[0]:(rank + 1) * self.word_embeddings.weight.shape[0]]
            assert weight.shape == self.word_embeddings.weight.shape, f"{weight.shape} != {self.word_embeddings.weight.shape}"
            self.word_embeddings.weight.data = weight
        
        if args.embedding_checkpoint is not None:
            rank = mpu.get_tensor_model_parallel_rank()
            path = f"{args.embedding_checkpoint}/layer_01-model_0{rank}-model_states.pt"
            tensor = torch.load(path)["word_embeddings.weight"].to(self.word_embeddings.weight)
            assert tensor.shape == self.word_embeddings.weight.shape, f"{tensor.shape} != {self.word_embeddings.weight.shape}"
            self.word_embeddings.weight.data = tensor

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        # embeddings = self.rms_norm(embeddings)
        return embeddings


class LlamaEmbeddingPipe(LlamaEmbedding):

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
        else:
            input_ids = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        attention_mask = inputs[2]

        # log inputs if on rank 0
        # if self.rank == 0:
        #     print("EMBEDDING INPUTS")
        #     input_ids_cpu = input_ids.cpu()
        #     for i in range(input_ids.size(1)):
        #         print(input_ids_cpu[0, i].long().item(), end=", ")
        #     print("\\ \n")
        #     attention_mask_cpu = attention_mask.cpu()
        #     for i in range(500):
        #         for j in range(500):
        #             print(attention_mask_cpu[0, 0, i, j].long().item(), end="")
        #         print(',')
        #     print(flush=True)

        embeddings = super().forward(input_ids)

        return embeddings, attention_mask


class LlamaLMHead(MegatronModule):
    """Causal LM head for Llama

    Arguments:
        vocab_size: size of vocabulary.
        hidden_size: hidden size
        gather_output: wether output logits being gathered or not.
        init_method: init method for weight initialization
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 init_method,
                 parallel_output=True,
                 hf_checkpoint=None):
        super(LlamaLMHead, self).__init__()
        args = get_args()
        self.hidden_size = hidden_size
        self.init_method = init_method_normal(0.0001)  # https://aclanthology.org/W17-7508/

        self.parallel_output = parallel_output

        self.lm_head = mpu.ColumnParallelLinear(input_size=self.hidden_size,
                                                output_size=vocab_size,
                                                bias=False,
                                                gather_output=not self.parallel_output,
                                                skip_bias_add=True,
                                                init_method=init_method  # _normal_l2(0.2676)
        )

        if hf_checkpoint is not None and args.embedding_checkpoint is None:
            rank = mpu.get_tensor_model_parallel_rank()
            weight = hf_checkpoint.get_tensor("lm_head.weight")
            weight = weight[rank * self.lm_head.weight.shape[0]:(rank + 1) * self.lm_head.weight.shape[0]]
            assert weight.shape == self.lm_head.weight.shape, f"{weight.shape} != {self.lm_head.weight.shape}"
            self.lm_head.weight.data = weight
        
        if args.embedding_checkpoint is not None:
            rank = mpu.get_tensor_model_parallel_rank()
            path = f"{args.embedding_checkpoint}/layer_45-model_0{rank}-model_states.pt"
            tensor = torch.load(path)["lm_head.weight"].to(self.lm_head.weight)
            assert tensor.shape == self.lm_head.weight.shape, f"{tensor.shape} != {self.lm_head.weight.shape}"
            self.lm_head.weight.data = tensor
        

    def forward(self, inputs):
        logits, _ = self.lm_head(inputs)
        return logits


class LlamaLMHeadPipe(LlamaLMHead):

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        logits_parallel = super().forward(hidden_states)

        if self.parallel_output:
            return logits_parallel

        return mpu.gather_from_tensor_model_parallel_region(logits_parallel)


class MistralModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(
        self,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        prefix_lm=False,
    ):
        super(MistralModel, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            # TODO: Change naming of class from GPT to something that encapsulate prefix lm.
            encoder_attn_mask_type=AttnMaskType.prefix if prefix_lm else AttnMaskType.causal,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers),
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.initialize_word_embeddings(init_method_normal)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None, curriculum_seqlen=None):
        if curriculum_seqlen is not None:
            args = get_args()
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value)

        if self.post_process:
            return post_language_model_processing(
                lm_output, labels,
                self.word_embeddings_weight(),
                get_key_value,
                self.parallel_output,
                forward_method_parallel_output,
                self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


def get_cross_entropy(is_prefix: bool):
    def CrossEntropy(output, labels):
        labels, loss_mask = labels[0], labels[1]

        args = get_args()

        # if args.rank < 4:
        #     print("CROSS ENTROPY INPUTS")
        #     labels_cpu = labels.cpu()
        #     for i in range(labels.size(1)):
        #         print(labels_cpu[0, i].long().item(), end=", ")
        #     print("\\ \n")
        #     loss_mask_cpu = loss_mask.cpu()
        #     for i in range(labels.size(1)):
        #         print(loss_mask_cpu[0, i].long().item(), end=", ")
        #         print('\\ \n')
        #     print(flush=True)

        losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)

        if is_prefix:
            micro_batch_size, sequence_length = loss_mask.shape
            average_tokens_per_sample: torch.Tensor
            if args.loss_on_targets_only:
                # HACK: This is useful when we obtain loss masks that are microbatch dependent. Consequently, if we want to
                #   preserve the notion that all tokens have the same impact on the loss, we can only normalise using a
                #   microbatch independent value. It should be expected weight over a microbatch.
                #   Here we still use `sequence_length`, that's batch size dependent, in order to be backwards compatible with
                #   current experiment on vanilla gpt.
                if args.reweight_loss_based_on_position_frequency:
                    reweight = torch.arange(
                        sequence_length, 0, -1, dtype=torch.float, device=loss_mask.device
                    ) / (sequence_length + 1) * 2
                    average_tokens_per_sample = reweight.flip(-1).cumsum(-1).mean()
                else:
                    average_tokens_per_sample = (sequence_length + 1) / 2
            else:
                average_tokens_per_sample = sequence_length
            expected_number_of_tokens = average_tokens_per_sample * micro_batch_size
        else:
            expected_number_of_tokens = loss_mask.sum()

        loss_mask = loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / expected_number_of_tokens.clamp(min=1.0)

        return loss

    return CrossEntropy


class MistralModelPipe(PipelineModule,MegatronModule):
    """Mistral Language model."""

    def __init__(
        self,
        num_tokentypes=0,
        parallel_output=True,
        attn_mask_type: AttnMaskType = AttnMaskType.custom
    ):
        args = get_args()
        self.parallel_output = parallel_output

        #init_method = init_method_normal(0.0033)
        init_method = init_method_normal(0.01397542485)

        self.specs = []

        if args.hf_checkpoint is not None:
            hf_checkpoint = safe_open(args.hf_checkpoint, framework="pt", device="cpu")
        else:
            hf_checkpoint = None

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        self.specs.append(
            LayerSpec(
                LlamaEmbeddingPipe, 
                hidden_size=args.hidden_size, vocab_size=args.padded_vocab_size, init_method=init_method,
                hf_checkpoint=hf_checkpoint
            )
        )

        if args.fp32_residual_connection:
            self.specs.append(lambda x: (x[0].transpose(0, 1).contiguous().float(), *x[1:]))
        else:
            self.specs.append(lambda x: (x[0].transpose(0, 1).contiguous(), *x[1:]))

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                    init_method=init_method,
                    output_layer_init_method=init_method,
                    layer_number=layer_idx,
                    # TODO: Change naming of class from GPT to something that encapsulate prefix lm.
                    self_attn_mask_type=attn_mask_type,
                    freeze=args.freeze,
                    hf_checkpoint=hf_checkpoint))

        # Undo data format change
        def undo(x):
            x = x[0]
            return x.transpose(0, 1).contiguous()
        self.specs.append(undo)

        if hf_checkpoint is not None:
            last_norm_checkpoint = hf_checkpoint.get_tensor("model.norm.weight")
        else:
            last_norm_checkpoint = None

        if args.transformer_checkpoint is not None:
            rank = mpu.get_tensor_model_parallel_rank()
            path = f"{args.transformer_checkpoint}/layer_{args.num_layers + 4}-model_0{rank}-model_states.pt"
            last_norm_checkpoint = torch.load(path)["weight"]

        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(RMSNorm,
                      args.hidden_size,
                      eps=args.layernorm_epsilon,
                      freeze=args.freeze,
                      hf_checkpoint=last_norm_checkpoint))

        self.specs.append(
            LayerSpec(LlamaLMHeadPipe, hidden_size=args.hidden_size, vocab_size=args.padded_vocab_size,
                      init_method=init_method, parallel_output=self.parallel_output,
                      hf_checkpoint=hf_checkpoint
            )
        )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        # here one can extend the regex to include more layers to be counted towards partitioning,
        # e.g. 'type:transformer|embedding' will add up all the transformer blocks and also the first
        # and last embedding layers and then partition that transformers+2 layers - so to get a good
        # balance you may want to use less transformer layers
        #
        # caveat emptor: the current implementation of PP fails unless each stage has at least one
        # transformer layer
        if args.pp_partition_method is not None:
            partition_method = args.pp_partition_method
        else:
            partition_method = 'type:transformer'

        super().__init__(layers=self.specs,
                         loss_fn=get_cross_entropy(is_prefix=attn_mask_type is AttnMaskType.prefix),
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method=partition_method)
