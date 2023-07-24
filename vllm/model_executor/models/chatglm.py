""" PyTorch ChatGLM model. """

from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm
from torch.nn.utils import skip_init
from transformers.utils import logging

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_rank
from vllm.model_executor.weight_utils import load_tensor_parallel_weights, hf_model_weights_iterator
from vllm.sequence import SequenceOutputs

logger = logging.get_logger(__name__)
KVCache = Tuple[torch.Tensor, torch.Tensor]


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 layer_id, hidden_size_per_attention_head=None, bias=True,
                 params_dtype=torch.float, position_encoding_2d=True, empty_init=True):
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        super(SelfAttention, self).__init__()

        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads
        self.position_encoding_2d = position_encoding_2d
        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // (self.num_attention_heads * 2)
            if position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            base=10000,
            precision=torch.half,
            learnable=False,
        )

        self.scale_mask_softmax = None

        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head

        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head

        # Strided linear layer.
        self.query_key_value = init_method(
            torch.nn.Linear,
            hidden_size,
            3 * self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )

        self.dense = init_method(
            torch.nn.Linear,
            self.inner_hidden_size,
            hidden_size,
            bias=bias,
            dtype=params_dtype,
        )

        # self.num_heads = total_num_heads // tensor_model_parallel_world_size
        total_num_heads = self.num_attention_heads
        head_dim = self.hidden_size // total_num_heads
        scale = head_dim ** -0.5
        self.attn = PagedAttention(
            total_num_heads,
            head_dim,
            scale=scale)

    def split_tensor_along_last_dim(self, tensor, num_partitions,
                                    contiguous_split_chunks=False):
        """Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                    in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
            cache_event: Optional[torch.cuda.Event],
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """
        hidden_states = torch.stack([hidden_states]).transpose(0, 1)
        positions = torch.stack([positions]).transpose(0, 1)

        # [seq_len, batch, 3 * hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)

        # [seq_len, batch, 3 * hidden_size] --> [seq_len, batch, num_attention_heads, 3 * hidden_size_per_attention_head]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

        if self.position_encoding_2d:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=positions.max() + 1)
            position_ids, block_position_ids = positions[0, :, :].contiguous(), \
                                               positions[1, :, :].contiguous()
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
        else:
            position_ids = positions
            cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
            # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)

        key_cache, value_cache = kv_cache
        assert query_layer.size(1) == key_layer.size(1) == value_layer.size(1) == 1
        query_layer = query_layer.transpose(0, 1)[0]
        key_layer = key_layer.transpose(0, 1)[0]
        value_layer = value_layer.transpose(0, 1)[0]
        context_layer = self.attn(query_layer, key_layer, value_layer, key_cache, value_cache,
                                  input_metadata, cache_event)

        output = self.dense(context_layer)
        return output


class GEGLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_fn = F.gelu

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class GLU(torch.nn.Module):
    def __init__(self, hidden_size, inner_hidden_size=None,
                 layer_id=None, bias=True, activation_func=gelu, params_dtype=torch.float, empty_init=True):
        super(GLU, self).__init__()
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        self.layer_id = layer_id
        self.activation_func = activation_func

        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = init_method(
            torch.nn.Linear,
            self.hidden_size,
            self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = init_method(
            torch.nn.Linear,
            self.inner_hidden_size,
            self.hidden_size,
            bias=bias,
            dtype=params_dtype,
        )

    def forward(self, hidden_states):
        """
        hidden_states: [seq_len, batch, hidden_size]
        """

        # [seq_len, batch, inner_hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)

        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class GLMBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            layernorm_epsilon,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            layernorm=LayerNorm,
            use_bias=True,
            params_dtype=torch.float,
            num_layers=28,
            position_encoding_2d=True,
            empty_init=True
    ):
        super(GLMBlock, self).__init__()
        # Set output layer initialization if not provided.

        self.layer_id = layer_id

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        self.position_encoding_2d = position_encoding_2d

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            bias=use_bias,
            params_dtype=params_dtype,
            position_encoding_2d=self.position_encoding_2d,
            empty_init=empty_init
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        self.num_layers = num_layers

        # GLU
        self.mlp = GLU(
            hidden_size,
            inner_hidden_size=inner_hidden_size,
            bias=use_bias,
            layer_id=layer_id,
            params_dtype=params_dtype,
            empty_init=empty_init
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
            cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """

        # Layer norm at the begining of the transformer layer.
        # [seq_len, batch, hidden_size]
        attention_input = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = self.attention(
            attention_input,
            positions=positions,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        return output  # hidden_states, present, attentions


class RealChatGLMModel(nn.Module):

    def __init__(self, config, empty_init=True):
        super().__init__()
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        # recording parameters
        self.max_sequence_length = config.max_sequence_length
        self.hidden_size = config.hidden_size
        self.params_dtype = torch.half
        self.num_attention_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.layernorm_epsilon = config.layernorm_epsilon
        self.inner_hidden_size = config.inner_hidden_size
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.position_encoding_2d = config.position_encoding_2d
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection

        self.config = config

        self.word_embeddings = init_method(
            torch.nn.Embedding,
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size,
            dtype=self.params_dtype
        )
        self.gradient_checkpointing = False

        def get_layer(layer_id):
            return GLMBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.layernorm_epsilon,
                layer_id,
                inner_hidden_size=self.inner_hidden_size,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                layernorm=LayerNorm,
                use_bias=True,
                params_dtype=self.params_dtype,
                position_encoding_2d=self.position_encoding_2d,
                empty_init=empty_init
            )

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(self.num_layers)]
        )

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)

    def get_position_ids(self, input_ids, mask_positions, device, use_gmasks=None):
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        if self.position_encoding_2d:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [torch.cat((
                torch.zeros(context_length, dtype=torch.long, device=device),
                torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
            )) for context_length in context_lengths]
            block_position_ids = torch.stack(block_position_ids, dim=0)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[i, context_length:] = mask_positions[i]

        return position_ids

    def build_positions(self, input_ids: torch.Tensor):
        assert len(input_ids.size()) == 1
        ids_list = input_ids.tolist()
        if not (ids_list[-2] == self.config.gmask_token_id and ids_list[-1] == self.config.bos_token_id):
            ids_list[-2], ids_list[-1] = self.config.gmask_token_id, self.config.bos_token_id
        input_ids = torch.LongTensor([ids_list]).to(input_ids.device)

        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        seqs = input_ids.tolist()

        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        position_ids = self.get_position_ids(
            input_ids,
            mask_positions=mask_positions,
            device=input_ids.device,
            use_gmasks=use_gmasks
        )
        return input_ids[0], position_ids[0]

    def forward(
            self,
            input_ids: torch.LongTensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        if positions is None:
            input_ids, positions = self.build_positions(input_ids)
        hidden_states = self.word_embeddings(input_ids)
        # hidden_states = inputs_embeds.transpose(0, 1)
        for i, layer in enumerate(self.layers):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            hidden_states = layer(
                hidden_states,
                positions=positions,
                kv_cache=kv_caches[i],
                input_metadata=input_metadata,
                cache_event=cache_event,
            )
        # Final layer norm.
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class ChatGLMModel(nn.Module):
    def __init__(self, config, empty_init=True):
        super().__init__()
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        self.max_sequence_length = config.max_sequence_length
        self.position_encoding_2d = config.position_encoding_2d
        self.transformer = RealChatGLMModel(config, empty_init=empty_init)
        self.lm_head = init_method(
            nn.Linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=torch.half
        )
        self.config = config
        self.sampler = Sampler(config.vocab_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        assert len(input_ids.size()) == 1
        hidden_states = self.transformer(
            input_ids=input_ids,
            positions=positions if len(positions.size()) == 2 and positions.size(0) == 2 else None,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )

        # hidden_states.transpose(0, 1)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = []
    _row_parallel_weights = []

    def load_weights(
            self,
            model_name_or_path: str,
            cache_dir: Optional[str] = None,
            use_np_cache: bool = False,
    ):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache
        ):
            if "rotary_emb.inv_freq" in name:
                continue

            param = state_dict[name]
            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                self._column_parallel_weights,
                self._row_parallel_weights,
                tensor_model_parallel_rank,
            )
