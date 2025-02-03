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

"""Transformer."""
import math
import torch
import torch.nn.functional as F
from torch import nn

from megatron import get_args, logging
from megatron import mpu
from .module import MegatronModule
from megatron.enums import AttnMaskType, LayerType, AttnType, PositionEmbeddingType
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.fused_rms_norm import MixedFusedRMSNorm as RMSNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu

import deepspeed

from .glu_activations import GLU_ACTIVATIONS
from .positional_embeddings import RotaryEmbedding, apply_rotary_pos_emb_torch, apply_rotary_pos_emb

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method, layer_number=0, hf_checkpoint=None):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to ffn_hidden_size
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            # GLU is a special activation that divides the dimension by a factor 2.
            2 * args.ffn_hidden_size if args.glu_activation else args.ffn_hidden_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        n_tp = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()

        if hf_checkpoint is not None:
            tensor_up = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.mlp.up_proj.weight").chunk(n_tp, 0)[tp_rank].to(self.dense_h_to_4h.weight)
            tensor_gate = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.mlp.gate_proj.weight").chunk(n_tp, 0)[tp_rank].to(self.dense_h_to_4h.weight)
            tensor = torch.cat([tensor_up, tensor_gate], dim=0)
            assert tensor.shape == self.dense_h_to_4h.weight.shape, f"{tensor.shape} != {self.dense_h_to_4h.weight.shape}"
            self.dense_h_to_4h.weight.data = tensor

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.silu
        if args.glu_activation:
            self.activation_func = GLU_ACTIVATIONS[args.glu_activation]
        elif args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

        if hf_checkpoint is not None:
            tensor = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.mlp.down_proj.weight").chunk(n_tp, 1)[tp_rank].to(self.dense_4h_to_h.weight)
            assert tensor.shape == self.dense_4h_to_h.weight.shape, f"{tensor.shape} != {self.dense_4h_to_h.weight.shape}"
            self.dense_4h_to_h.weight.data = tensor

        if args.transformer_checkpoint is not None:
            rank = mpu.get_tensor_model_parallel_rank()
            path = f"{args.transformer_checkpoint}/layer_{layer_number + 3:02d}-model_0{rank}-model_states.pt"
            checkpoint_dict = torch.load(path)
            tensor_up = checkpoint_dict["mlp.dense_h_to_4h.weight"].to(self.dense_h_to_4h.weight)
            tensor_down = checkpoint_dict["mlp.dense_4h_to_h.weight"].to(self.dense_4h_to_h.weight)
            self.dense_h_to_4h.weight.data = tensor_up
            self.dense_4h_to_h.weight.data = tensor_down


    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding, hf_checkpoint=None):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.rank = args.rank
        self.fp16 = args.fp16
        self.bf16 = args.bf16
        self.position_embedding_type = args.position_embedding_type

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type

        q_projection_size = args.kv_channels * args.num_attention_heads
        kv_projection_size = args.kv_channels * args.num_kv_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(q_projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            q_projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        self.num_kv_attention_heads = mpu.divide(
            args.num_kv_attention_heads, world_size)
        self.kv_repetitions =  self.num_attention_heads_per_partition // self.num_kv_attention_heads

        self.query = mpu.ColumnParallelLinear(
            args.hidden_size,
            q_projection_size,
            bias=False,
            gather_output=False,
            init_method=init_method
        )

        self.key_value = mpu.ColumnParallelLinear(
            args.hidden_size,
            2 * kv_projection_size,
            bias=False,
            gather_output=False,
            init_method=init_method
        )

        n_tp = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()

        if hf_checkpoint is not None:
            tensor_k = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.self_attn.k_proj.weight").chunk(n_tp, 0)[tp_rank].to(self.key_value.weight)
            tensor_q = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.self_attn.q_proj.weight").chunk(n_tp, 0)[tp_rank].to(self.key_value.weight)
            tensor_v = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.self_attn.v_proj.weight").chunk(n_tp, 0)[tp_rank].to(self.key_value.weight)
            tensor_kv = torch.cat([tensor_k, tensor_v], dim=0)
            assert tensor_kv.shape == self.key_value.weight.shape, f"{tensor_kv.shape} != {self.key_value.weight.shape}"
            self.key_value.weight.data = tensor_kv
            assert tensor_q.shape == self.query.weight.shape, f"{tensor_q.shape} != {self.query.weight.shape}"
            self.query.weight.data = tensor_q

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            q_projection_size,
            args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=output_layer_init_method)

        if hf_checkpoint is not None:
            tensor = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.self_attn.o_proj.weight").chunk(n_tp, 1)[tp_rank].to(self.dense.weight)
            assert tensor.shape == self.dense.weight.shape, f"{tensor.shape} != {self.dense.weight.shape}"
            self.dense.weight.data = tensor

        if args.transformer_checkpoint is not None:
            rank = mpu.get_tensor_model_parallel_rank()
            path = f"{args.transformer_checkpoint}/layer_{layer_number + 3:02d}-model_0{rank}-model_states.pt"
            checkpoint_dict = torch.load(path)
            tensor_q = checkpoint_dict["self_attention.query.weight"].to(self.query.weight)
            tensor_kv = checkpoint_dict["self_attention.key_value.weight"].to(self.key_value.weight)
            tensor_o = checkpoint_dict["self_attention.dense.weight"].to(self.dense.weight)
            self.query.weight.data = tensor_q
            self.key_value.weight.data = tensor_kv
            self.dense.weight.data = tensor_o

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

        if self.position_embedding_type == PositionEmbeddingType.rotary:
            self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head)

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # log inputs if on rank 0
        # if self.rank == 0:
        #     print(f"ATTENTION LAYER {self.layer_number}")
        #     attention_mask_cpu = attention_mask.cpu()
        #     for i in range(attention_mask.size(2)):
        #         for j in range(attention_mask.size(3)):
        #             print(attention_mask_cpu[0, 0, i, j].long().item(), end="")
        #         print(",")
        #     print(flush=True)

        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(hidden_states)
        query_layer, _ = self.query(hidden_states)

        key_layer, value_layer = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Changed layout, for compatibility with CKPT conversion
        new_query_shape = query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        new_key_shape = key_layer.size()[:-1] + (self.num_kv_attention_heads, self.hidden_size_per_attention_head)

        query_layer = query_layer.view(new_query_shape)
        key_layer = key_layer.view(new_key_shape)
        value_layer = value_layer.view(new_key_shape)

        key_layer = torch.repeat_interleave(key_layer, self.kv_repetitions, dim=-2)
        value_layer = torch.repeat_interleave(value_layer, self.kv_repetitions, dim=-2)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape(output_size[3],  output_size[0] * output_size[1], -1)

        # Rotary embeddings
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            apply_rotary_fn = apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb

            seq_len = key_layer.shape[0]
            offset = 0
            if layer_past is not None and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.bmm(
            query_layer.transpose(0, 1) * (1.0 / self.norm_factor),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2)  # [b * np, hn, sk]
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                # TODO @thomasw21 Handle case where `attention_mask` is None
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, _ = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding, freeze=False, hf_checkpoint=None):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        self.input_layernorm = RMSNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if hf_checkpoint is not None:
            tensor = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.input_layernorm.weight").to(self.input_layernorm.weight)
            assert tensor.shape == self.input_layernorm.weight.shape, f"{tensor.shape} != {self.input_layernorm.weight.shape}"
            self.input_layernorm.weight.data = tensor

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type,
            hf_checkpoint=hf_checkpoint)
        self.hidden_dropout = 0.0
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = RMSNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if hf_checkpoint is not None:
            tensor = hf_checkpoint.get_tensor(f"model.layers.{layer_number}.post_attention_layernorm.weight").to(self.post_attention_layernorm.weight)
            assert tensor.shape == self.post_attention_layernorm.weight.shape, f"{tensor.shape} != {self.post_attention_layernorm.weight.shape}"
            self.post_attention_layernorm.weight.data = tensor

        if args.transformer_checkpoint is not None:
            rank = mpu.get_tensor_model_parallel_rank()
            path = f"{args.transformer_checkpoint}/layer_{layer_number + 3:02d}-model_0{rank}-model_states.pt"
            checkpoint = torch.load(path)
            tensor_input = checkpoint["input_layernorm.weight"].to(self.input_layernorm.weight)
            tensor_output = checkpoint["post_attention_layernorm.weight"].to(self.post_attention_layernorm.weight)
            self.input_layernorm.weight.data = tensor_input
            self.post_attention_layernorm.weight.data = tensor_output
        
        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        # MLP
        self.mlp = ParallelMLP(
            init_method,
            output_layer_init_method,
            layer_number=layer_number,
            hf_checkpoint=hf_checkpoint
        )

        # don't track gradient of the whole layer
        self.freeze = freeze

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                layer_past=None, get_key_value=False):
        # hidden_states: [b, s, h]

        # don't track gradient of the whole layer
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = \
            self.self_attention(layernorm_output,
                                attention_mask,
                                layer_past=layer_past,
                                get_key_value=get_key_value)

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = attention_output + residual

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            # re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, _ = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = mlp_output + residual

        if get_key_value:
            output = [output, presents]

        return output


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.
    
    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.
    """
    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            hidden_states, attention_mask = inputs, None
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')
        
    @property
    def input_layernorm_weight(self):
        return self.input_layernorm.weight.data
    
    @property
    def self_attention_key_value_weight(self):
        return self.self_attention.key_value.weight.data

    @property
    def self_attention_query_weight(self):
        return self.self_attention.query.weight.data
    
    @property
    def self_attention_dense_weight(self):
        return self.self_attention.dense.weight.data
    
    @property
    def post_attention_layernorm_weight(self):
        return self.post_attention_layernorm.weight.data
    
    @property
    def mlp_dense_h_to_4h_weight(self):
        return self.mlp.dense_h_to_4h.weight.data

    @property
    def mlp_dense_4h_to_h_weight(self):
        return self.mlp.dense_4h_to_h.weight.data


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 pre_process=True, post_process=True):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers.
        assert args.num_layers % mpu.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // mpu.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type)
        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)
                return x_
            return custom_forward

        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, enc_dec_attn_mask=None):

        # Checks.
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        if self.pre_process:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            # If the input flag for fp32 residual connection is set, convert for float.
            if self.fp32_residual_connection:
                hidden_states = hidden_states.transpose(0, 1).contiguous().float()
            # Otherwise, leave it as is.
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        if encoder_output is not None:
             encoder_output = encoder_output.transpose(0, 1).contiguous()

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask,
                                                       encoder_output,
                                                       enc_dec_attn_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      encoder_output=encoder_output,
                                      enc_dec_attn_mask=enc_dec_attn_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        # Final layer norm.
        if self.post_process:
            # Reverting data format change [s b h] --> [b s h].
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states
        if get_key_value:
            output = [output, presents]

        return output
