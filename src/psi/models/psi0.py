import os
import copy
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, Dict, Any
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from PIL import Image
from dataclasses import dataclass
from diffusers.utils.outputs import BaseOutput
from diffusers.configuration_utils import ConfigMixin, FrozenDict, register_to_config
# from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_utils import ModelMixin
# from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
# from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero, AdaLayerNormContinuous
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.attention import FeedForward, _chunked_feed_forward
# from dit_policy.data4robotics.models.diffusion import _TimeNetwork
from psi.config.config import LaunchConfig
from transformers import Qwen3VLForConditionalGeneration, AutoConfig, AutoProcessor, Qwen2TokenizerFast, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

from psi.utils import initialize_overwatch, count_parameters
from psi.utils.attention import qwen3vl_attn_implementation

# from InternVLA.model.modules.action_model.DiT_modules.models import DiT
# from InternVLA.model.modules.projector.QFormer import CrossAttentionBlock
overwatch = initialize_overwatch(__name__)

# from psi.config.model import ModelConfig, InternVLA_M1_ModelConfig
# from psi.learn.vlms import _QWen_VL_Interface
from transformers.models.siglip import SiglipModel
from transformers.models.dinov2 import Dinov2Model
# from vlt.transformers.vla import ActionTransformerModel

from diffusers.utils import logging
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name  FIXME why this not works any more?

QWEN3VL_VARIANT = "Qwen/Qwen3-VL-2B-Instruct"

@dataclass
class HumanFoundationModelOutput(BaseOutput):
    action: "torch.Tensor"  # noqa: F821
    # Mean per-token L2 norm of the VLM feature fed to the action header (for logging).
    vlm_feat_norm: Optional["torch.Tensor"] = None  # noqa: F821

    def to_tuple(self):
        return (None, self.action)

@dataclass
class ActionTransformerModelOutput(BaseOutput):
    action: "torch.Tensor"  # noqa: F821

    def to_tuple(self):
        return (None, self.action)

class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        # self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))
        self.w = nn.Parameter(w, requires_grad=learnable_w)

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # assert len(x.shape) == 1, "assumes 1d input timestep array"
        # RTC: support 2D timesteps (B,Tp)
        x = x[..., None] * self.w
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        return self.out_net(x)

class CombinedTimestepTextProjEmbeddingsND(CombinedTimestepTextProjEmbeddings):
    """`combined_temb` embedding that also accepts a per-token timestep `(B, Tp)` (RTC).

    diffusers' `time_proj` (get_timestep_embedding) is 1-D only, so a `(B, Tp)` timestep
    would crash the base class. Here we flatten -> embed -> reshape and add the pooled
    (per-sample) instruction projection broadcast over the token axis:

        temb[b, i] = timestep_embed(t[b, i]) + pooled_proj(x[b])

    A 1-D `(B,)` timestep reproduces the base class exactly. Submodule names/params are
    unchanged, so existing combined_temb checkpoints load as-is.
    """

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        pooled = self.text_embedder(pooled_projection)  # (B, D)
        if timestep.dim() == 1:                          # (B,) -> (B, D)  [base-class path]
            proj = self.time_proj(timestep)
            temb = self.timestep_embedder(proj.to(dtype=pooled_projection.dtype))
            return temb + pooled
        elif timestep.dim() == 2:                        # (B, Tp) -> (B, Tp, D)  [RTC]
            B, T = timestep.shape
            proj = self.time_proj(timestep.reshape(B * T))
            temb = self.timestep_embedder(proj.to(dtype=pooled_projection.dtype))
            return temb.reshape(B, T, -1) + pooled.unsqueeze(1)
        raise ValueError(f"timestep must be 1-D or 2-D, got shape {tuple(timestep.shape)}")

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, int):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


if torch.__version__ >= "2.1.0":
    LayerNorm = nn.LayerNorm # type:ignore
else:
    # Has optional bias parameter compared to torch layer norm
    # TODO: replace with torch layernorm once min required torch version >= 2.1
    class LayerNorm(nn.Module):
        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()

            self.eps = eps

            if isinstance(dim, int):
                dim = (dim,)

            self.dim = torch.Size(dim)

            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)

class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        if len(conditioning_embedding.shape) == 2: # (B, D)
            conditioning_embedding = conditioning_embedding.unsqueeze(1) # (B, 1, D)
        elif len(conditioning_embedding.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of conditioning_embedding: {conditioning_embedding.shape}")
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x

class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
            assert False, "Not implemented with emb shape == 3, (B, T, D)"
        
        if len(emb.shape) == 2: # (B, D)
            emb = emb.unsqueeze(1) # (B, 1, D)
        elif len(emb.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of emb: {emb.shape}")
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer("pe", torch.clone(pe)) # <-- SONGLIN: this will cause problems when accelerate.prepare
        # self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()

class FilmConditioning(nn.Module):
    """Layer that adds FiLM conditioning.

    This is intended to be applied after a convolutional layer. It will learn a
    multiplicative and an additive factor to be applied to each channel of the
    convolution's output.

    Conv layer can be rank 2 or 4.

    For further details, see: https://arxiv.org/abs/1709.07871
    """
    
    def __init__(self,
                 in_dim: int,
                 num_channels: int):
        """Constructs a FiLM conditioning layer.

        Args:
            num_channels: Number of filter channels to expect in the input.
        """
        super(FilmConditioning, self).__init__() 
        # Note that we initialize with zeros because empirically we have found
        # this works better than initializing with glorot.
        self._projection_add = nn.Linear(in_dim, num_channels)
        self._projection_mult = nn.Linear(in_dim, num_channels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, conv_filters, conditioning):
        assert len(conditioning.shape) == 2
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)

        if len(conv_filters.shape) == 4:
            # [B, D] -> [B, D, 1, 1]
            projected_cond_add = projected_cond_add.unsqueeze(-1).unsqueeze(-1)
            projected_cond_mult = projected_cond_mult.unsqueeze(-1).unsqueeze(-1)
        elif len(conv_filters.shape) == 3:
            # [B, D] -> [B, 1, D]
            projected_cond_add = projected_cond_add.unsqueeze(1)
            projected_cond_mult = projected_cond_mult.unsqueeze(1)
        else:
            assert len(conv_filters.shape) == 2
        
        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result

class JointVLAAttnProcessor: #(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        # self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        # self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        # self.out_dim = out_dim if out_dim is not None else query_dim

        # self.to_q_lang = nn.Linear(query_dim, self.inner_dim, bias=bias)
        # self.to_k_lang = nn.Linear(query_dim, self.inner_kv_dim, bias=bias)
        # self.to_v_lang = nn.Linear(query_dim, self.inner_kv_dim, bias=bias)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)

        assert attn.to_k is not None
        key = attn.to_k(hidden_states)

        assert attn.to_v is not None
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            if attention_mask is not None:
                assert attention_mask.dtype == torch.float32
                attn_mask = torch.cat([
                    torch.ones(hidden_states.shape[0], 1, 1, hidden_states.shape[1], device=attention_mask.device).to(torch.bool),  # (B, 1, 1, S1)
                    (attention_mask == 1)[:, None, None, :]  # (B, 1, 1, S2)
                ], dim=-1)
            else:
                attn_mask = None

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask) # type: ignore
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # type: ignore
        hidden_states = hidden_states.to(query.dtype) # type: ignore
 
        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        ) # type: ignore
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, encoder_hidden_states
    
        
class ObservationProjection(nn.Module): # FIXME naming
    """ VLT observation projector

    Args:
        ...
    """

    def __init__(
        self,
        action_pred_horizon: int = 50, # Tp
        action_dim: int = 7, # in dim
        output_dim: int = 512, # out dim
        hidden_dim: int = 1280,
        n_conditions: int = 1, # num image of contitions
        timestep_in_dim: int = 320,
        timestep_flip_sin_to_cos: bool = True,
        timestep_freq_shift: int = 0,
        token_fusion: str = "concat",
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False
    ):
        super().__init__()

        # TODO 1. compare
        
        # self.time_net = _TimeNetwork(time_dim=256, out_dim=512)
        
        # self.time_proj = Timesteps(timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift)
        # self.time_embedding = TimestepEmbedding(timestep_in_dim, output_dim, act_fn="silu")
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim
        # self.ac_proj = nn.Sequential(
        #     nn.Linear(action_dim, action_dim),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(action_dim, output_dim),
        # )

        # _DiTNoiseNet.__init__
        self.enc_pos = _PositionalEncoding(d_model=output_dim)

        

        # agent.__init__
        n_cams = 0 # 1
        
        imgs_per_cam = 1
        self.camera_indices = [] # [0]
        self._n_cams = n_cams
        self._share_cam_features = False # model_cfg.share_cam_features
        early_fusion = False
        text_dim = 768
        # odim = 15
        self.token_fusion = token_fusion

        if self._share_cam_features:
            ...
        else:
            if n_conditions > 0:
                from dit_policy.data4robotics.models.resnet import ResNet
                features = ResNet(
                    size=18,
                    weights="IMAGENET1K_V1",
                    restore_path=resnet_store_path,
                    avg_pool=False,
                    conv_repeat=0,
                    norm_cfg={
                        "name": "group_norm",
                        "num_groups": 16,
                    }
                )
            else:
                features = None
            feat_list = [copy.deepcopy(features) for _ in range(0, n_conditions)] # + [copy.deepcopy(features) for _ in range(1, n_cams)] + [copy.deepcopy(features) for _ in range(0, n_conditions)]
            self.visual_features = nn.ModuleList(feat_list)
            self.views_proj = nn.Linear(view_feature_dim, output_dim, bias=True) # TODO configure 1920 # boqian_fix

            # TODO abl: more layers # view proj
            # self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            # self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            # self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            # self.act_fn1 = nn.GELU()
            # self.act_fn2 = nn.GELU()

            # self.views_proj = nn.Sequential(
            #     nn.Conv2d(16, output_dim, kernel_size=(2, 2), stride=2, bias=True),
            #     nn.ReLU(),
            #     nn.Conv2d(output_dim, output_dim, kernel_size=(2, 2), stride=2, bias=True)
            # )
            # self.traj2d_proj = nn.ModuleList([
            #     nn.Conv2d(16, 512, kernel_size=(2, 2), stride=2, bias=True),
            #     nn.ReLU(),
            #     nn.Conv2d(512, output_dim, kernel_size=(2, 2), stride=2, bias=True),
            #     nn.ReLU(),
            #     nn.Linear(output_dim, output_dim, bias=True),
            #     nn.LayerNorm(output_dim)
            # ])
        
        self.early_fusion = early_fusion
        self.imgs_per_cam = imgs_per_cam
        imgs_per_cam = 1 if early_fusion else imgs_per_cam
        self._n_conditions = n_conditions

        self.embed_proj = nn.Identity()
        if self.token_fusion != "perceiver" and self._n_conditions > 0:
            assert features is not None
            self._token_dim = features.embed_dim
            if output_dim != self._token_dim:
                self.embed_proj = nn.Linear(self._token_dim, output_dim, bias=False)
        else:
            self._token_dim = 0

        self._n_tokens = imgs_per_cam * n_cams * self._token_dim + n_conditions * self._token_dim
        self.traj_n_token = self._token_dim
        
        self.feat_norm = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6) # feat_norm = None
        self.feat_traj_norm = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6) # feat_norm = None
        # self.traj2d_proj = nn.ModuleList([
        #     nn.Linear(output_dim, output_dim) for _ in range(2)
        # ])
        # self.traj2d_proj = nn.Conv2d(
        #     16, output_dim, kernel_size=(2, 2), stride=2, bias=True
        # )

        if self.token_fusion == "cross":
            
            self.attn = Attention(
                query_dim=self._token_dim,
                cross_attention_dim=1920,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )
        elif self.token_fusion == "perceiver":
            num_queries = 256
            self.latents1 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) / hidden_dim**0.5)
            # self.proj_in1 = nn.Linear(1920, hidden_dim)  -> views_proj (see above)
            self.cross_attn1 = Attention(
                query_dim=hidden_dim,
                cross_attention_dim=hidden_dim,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )
            self.latents2 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) / hidden_dim**0.5)
            self.proj_in2 = nn.Linear(self._token_dim, hidden_dim)
            self.cross_attn2 = Attention(
                query_dim=hidden_dim,
                cross_attention_dim=hidden_dim,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )

        self.film = FilmConditioning(2048, output_dim) if use_film else None

        self._obs_strat = "add_token"
        self._n_tokens += 1

        self._obs_proc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(odim, output_dim)
        )
        dropout = 0.1 # @see model_cfg.dropout
        linear_proj = nn.Identity() # build (optional) token feature projection layer 
        norm = nn.Identity() # feat_norm = None
        self.post_proc = nn.Sequential(linear_proj, norm, nn.Dropout(dropout))

    def forward(self, 
                # noisy_actions: torch.Tensor, 
                # timestep: torch.Tensor, 
                # temp
                views, 
                obs, 
                traj2ds=None, 
                # ac_flat = None, 
                # mask_flat = None, 
                text_embeddings = None, 
                vlm_attn_mask=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            noisy_actions (`torch.Tensor`):
                Image features.
            timestep (`torch.Tensor`):
                Timestep in denoising process.
        Returns:
            `Tuple`[`torch.Tensor`, `torch.Tensor`]: The pair (latents, timestep_emb).
        """
        # DiffusionTransformerAgent.forward
        vlm_token_len = views.shape[1] * views.shape[2]  # V*S VLM tokens, right-padded
        s_t, out_attn_mask = self.tokenize_obs(views, obs, traj2ds, text_embeddings=text_embeddings, vlm_attn_mask=vlm_attn_mask) # (B, S, d_model)
        # _DiTNoiseNet.forward_enc
        s_t = s_t.transpose(0, 1) # (S, B, d_model)
        s_t = self._add_obs_pos(s_t, vlm_token_len) # (S, B, d_model)
        return s_t.transpose(0,1), out_attn_mask # (B,S,d_model)

    def _add_obs_pos(self, s_t, vlm_token_len):
        """Positional encoding for the [VLM tokens ..., obs token(s)] sequence.
            Only the observation tokens receive positional encodings. 
            The VLM tokens retain their original positions as determined by the VLM's own positional encoding.
        """
        n_obs = s_t.shape[0] - vlm_token_len
        if n_obs <= 0:
            return s_t
        obs_pos = self.enc_pos.pe[:n_obs].to(s_t.dtype)  # (n_obs, 1, d_model)
        return torch.cat([s_t[:vlm_token_len], s_t[vlm_token_len:] + obs_pos], dim=0)

    def tokenize_obs(self, views, obs, traj2ds = None, flatten=False, text_embeddings=None, vlm_attn_mask=None):
        
        view_tokens = self.views_proj(views)
        B,V,S,D = view_tokens.shape
        
        if traj2ds is not None:
            traj_tokens = self.embed({}, {f"cond{i}": traj2ds[:, i] for i in range(traj2ds.shape[1])})

            if self.token_fusion == "concat":
                traj_tokens = self.embed_proj(traj_tokens) #(B,S,d_model)
                tokens = torch.cat([view_tokens.view(B,V*S,D), traj_tokens], dim=1) # (B, tok1+tok2, 1536)
            elif self.token_fusion == "cross":  
                traj_tokens_list, view_tokens_list = [], []
                for v_idx in range(traj2ds.shape[1]):
                    # traj2d --> cross attend --> view features
                    view_tokens = self.attn(
                        hidden_states=traj_tokens[:, v_idx*self.traj_n_token, (v_idx+1)*self.traj_n_token],
                        encoder_hidden_states=views[:, v_idx],
                    )
                    traj_tokens_list.append(traj_tokens)
                    view_tokens_list.append(view_tokens)

                traj_tokens = torch.cat(traj_tokens_list, dim=1)
                traj_tokens = self.embed_proj(traj_tokens)
                view_tokens = torch.cat(view_tokens_list, dim=1)
                tokens = torch.cat([view_tokens, traj_tokens], dim=1) # (B, tok1+tok2, 1536)
            elif self.token_fusion == "perceiver":
                latents1 = self.latents1.repeat(B, 1, 1)
                # views = self.proj_in1(views)
                # views = views + text_embeddings[:, None]
                tokens1 = self.cross_attn1(
                    hidden_states=latents1,
                    encoder_hidden_states=view_tokens.view(B, V*S, D),
                )
                latents2 = self.latents2.repeat(B, 1, 1)
                traj_tokens = self.proj_in2(traj_tokens)
                # traj2ds = traj2ds + text_embeddings[:, None]
                tokens2 = self.cross_attn2(
                    hidden_states=latents2,
                    encoder_hidden_states=traj_tokens,
                )
                tokens = torch.cat([tokens1, tokens2], dim=1) # (B, tok1+tok2, 1536)
            else:
                raise ValueError
        else:
            assert self._n_conditions == 0, "inconsistent confg"
            tokens = view_tokens.view(B,V*S,D)

        if (self.film is not None) and (text_embeddings is not None):
            tokens = self.film(tokens, text_embeddings)

        if self._obs_strat == "add_token":
            obs_token = self._obs_proc(obs)#[:, None]
            tokens = torch.cat((tokens, obs_token), 1)
            obs_token_len = obs_token.shape[1]
            vlm_attn_mask = (
                torch.cat([vlm_attn_mask, torch.ones((B, obs_token_len), device=vlm_attn_mask.device)], 1) 
                if vlm_attn_mask is not None else None
            )
        elif self._obs_strat == "pad_img_tokens":
            obs = self._obs_proc(obs)
            obs = obs[:, None].repeat((1, tokens.shape[1], 1))
            tokens = torch.cat((obs, tokens), 2)
            vlm_attn_mask = None
        else:
            assert self._obs_strat is None

        tokens = self.post_proc(tokens)
        if flatten:
            return tokens.reshape((tokens.shape[0], -1)), vlm_attn_mask
        return tokens, vlm_attn_mask
    
    def embed(self, imgs, conditions=None):
        """ SONGLIN: DOWNSCALE == 2**5 = 32, total 5 conv2d with stride(2) """
        def embed_helper(net, im):
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # TODO check if this is needed
            if self.early_fusion and len(im.shape) == 5:
                T = im.shape[1]
                im = torch.cat([im[:, t] for t in range(T)], 1)
                embeds = net(im)
            elif len(im.shape) == 5:
                B, T, C, H, W = im.shape
                embeds = net(im.reshape((B * T, C, H, W)))
                embeds = embeds.reshape((B, -1, net.embed_dim))
            else:
                assert len(im.shape) == 4
                embeds = net(im)
            return embeds

        if self._share_cam_features:
            embeds = [
                embed_helper(self.visual_features, imgs[f"cam{i}"])
                for i in self.camera_indices
            ]
            if conditions is not None:
                assert self._n_conditions > 0
                embeds += [
                    embed_helper(self.visual_features, conditions[f"cond{i}"]) 
                    for i in range(self._n_conditions)
                ]
        else:
            assert conditions is not None
            # img_encoders = self.visual_features[:self._n_cams] # sum([v.sum().item() for k, v in img_encoders.named_parameters()])
            # cond_encoders = self.visual_features[self._n_cams:]
            cond_encoders = self.visual_features
            # [
            #     embed_helper(net, imgs[f"cam{i}"])
            #     for i, net in zip(self.camera_indices, img_encoders)
            # ] +
            embeds = [
                embed_helper(net, conditions[f"cond{0}"])
                for i, net in enumerate(cond_encoders)
            ]

        """ if self.training: # FIXME better handle this thing
            return torch.cat(embeds, dim=1) """
        assert conditions is not None
        in_dtype = next(iter(imgs.values())).dtype if len(imgs) > 0 else next(iter(conditions.values())).dtype
        embeds = torch.cat(embeds, dim=1).to(dtype=in_dtype) # (B, S, d_model)
        return embeds

class ActionProjectionIn(nn.Module):
    def __init__(
        self,
        action_pred_horizon,
        action_dim,
        output_dim,
    ):  
        super().__init__()
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim

        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, output_dim),
        )
        # self.time_net = _TimeNetwork(time_dim=256, out_dim=output_dim)

        # self.register_parameter(
        #     "dec_pos",
        #     nn.Parameter(torch.empty(action_pred_horizon, output_dim), requires_grad=True),
        # ) # (1, Tp, hidden_dim)
        self.dec_pos = nn.Parameter(torch.empty(action_pred_horizon, output_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.dec_pos.data)

    def forward(self, noisy_actions: torch.Tensor, 
        # timestep
        ) -> torch.Tensor:
        B = noisy_actions.shape[0]
        # action_dim = self._dit_obs_proj.action_dim
        # time_enc = self.time_net(timestep)
        noise_acs = noisy_actions.reshape(B, -1, self.action_dim) # DiffusionTransformerAgent.forward
        ac_tokens = self.ac_proj(noise_acs) # noisy_actions, _DiTNoiseNet.forward
        # ac_tokens = ac_tokens.transpose(0, 1) # (B, L, d_model) _DiTNoiseNet.forward_dec
        action_hidden_states = ac_tokens + self.dec_pos.unsqueeze(0)
        return action_hidden_states#, time_enc
    

class ActionProjectionOut(nn.Module): # _FinalLayer
    def __init__(self, hidden_size, action_dim, final_layer_norm: bool = True):
        super().__init__()
        self.final_layer_norm = final_layer_norm
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        if final_layer_norm:
            # DiT convention: zero-init the modulation so the head starts as a
            # plain LayerNorm (scale=shift=0 -> x*(1+0)+0) instead of a random
            # unbounded gate.
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
        # self.reset_parameters()

    def forward(self, x, t, cond=None):
        # process the conditioning vector first
        # cond = torch.mean(cond, dim=1)
        # cond = cond + t

        if len(t.shape) == 2: # (B, D)
            t = t.unsqueeze(1) # (B, 1, D)
        elif len(t.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of t: {t.shape}")
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1) # cond -> t
        if self.final_layer_norm:
            x = self.norm_final(x) * (1 + scale) + shift
        else:
            # Legacy head: the unnormalized residual stream times an unbounded
            # gate, with no identity path. d(loss)/dx scales with `scale` and
            # vice versa, so the two grow into each other -- this is what could drive
            # the grad-norm blowups.
            x = x * scale + shift
        x = self.linear(x)
        # return x.transpose(0, 1)
        return x
    

# @maybe_allow_in_graph
class VLATransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        training_phase: str = "joint",  # "joint", "action", "traj2d", "action_mini"
    ):
        super().__init__()

        # self.training_phase = training_phase

        # from models.dit_policy.data4robotics.models.diffusion import  _SelfAttnEncoder #_TransformerEncoder,
        # self.encoder_module = _SelfAttnEncoder(
        #     d_model=dim,
        #     nhead=num_attention_heads,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     activation="gelu",
        # )
        # self.encoder_module.reset_parameters() 

        # # decoder blocks
        # from models.dit_policy.data4robotics.models.diffusion import _DiTDecoder #_TransformerDecoder, 
        # self.decoder_module = _DiTDecoder(
        #     d_model=dim,
        #     nhead=num_attention_heads,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     activation="gelu",
        # )
        # self.decoder_module.reset_parameters() # _TransformerDecoder.__init__

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        # self.norm1 = AdaLayerNormZero(dim)
        self.norm1_act = AdaLayerNormZero(dim)
        logger.debug(f"AdaLN act parameters: {count_parameters(self.norm1_act):,}")
        if context_norm_type == "ada_norm_continous":
            """ self.norm1_lang = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm") """
            self.norm1_obs = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
            logger.debug(f"AdaLN obs parameters: {count_parameters(self.norm1_obs):,}")
        elif context_norm_type == "ada_norm_zero":
            """ self.norm1_lang = AdaLayerNormZero(dim) """
            self.norm1_obs = AdaLayerNormZero(dim)
            logger.debug(f"AdaLN obs parameters: {count_parameters(self.norm1_obs):,}")
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointVLAAttnProcessor(
                query_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                context_pre_only=context_pre_only,
            )
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor, # type: ignore
            qk_norm=qk_norm,
            eps=1e-6,
        )
        logger.debug(f"Joint Attention parameters: {count_parameters(self.attn):,}")

        self.norm2_act = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_act = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        logger.debug(f"FF act parameters: {count_parameters(self.ff_act):,}")

        if not context_pre_only:
            """ self.norm2_lang = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_lang = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate") """

            self.norm2_obs = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_obs = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
            logger.debug(f"FF obs parameters: {count_parameters(self.ff_obs):,}")
        else:
            """ self.norm2_lang = None
            self.ff_lang = None """

            self.norm2_obs= None
            self.ff_obs = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self, 
        # hidden_states: torch.FloatTensor,  #  V
        # lang_hidden_states: torch.FloatTensor,  # L
        action_hidden_states: torch.Tensor, # A
        obs_hidden_states: torch.Tensor, # O
        temb: torch.Tensor,
        obs_token_mask: Optional[torch.Tensor] = None,
        # obs_pos_emb: Optional[torch.FloatTensor] = None,
        # time_enc: Optional[torch.FloatTensor] = None,
        joint_attention_kwargs=None,
    ):
        # obs_hidden_states = self.encoder_module(src=obs_hidden_states, pos=obs_pos_emb)
        # action_hidden_states = self.decoder_module(x=action_hidden_states, t=time_enc, cond=obs_hidden_states) # recursive: _TransformerDecoder.forward
        
        norm_action_hidden_states, gate_msa_act, shift_mlp_act, scale_mlp_act, gate_mlp_act = self.norm1_act(action_hidden_states, emb=temb)

        if self.context_pre_only:
            """ norm_lang_hidden_states = self.norm1_lang(lang_hidden_states, temb) """
            norm_obs_hidden_states = self.norm1_obs(obs_hidden_states, temb[:,-1] if len(temb.shape) > 2 else temb)

            gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = None, None, None, None
        else:
            """ norm_lang_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_lang(
                lang_hidden_states, emb=temb
            ) """
            norm_obs_hidden_states, gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = self.norm1_obs(obs_hidden_states, emb=temb[:,-1] if len(temb.shape) > 2 else temb)

        act_attn_output, obs_attn_output = self.attn( ## indirectly calls JointAttnProcessor2_0.__call__
            # hidden_states=None,
            # lang_hidden_states=norm_lang_hidden_states,
            hidden_states=norm_action_hidden_states,
            encoder_hidden_states=norm_obs_hidden_states,
            attention_mask=obs_token_mask
            # **joint_attention_kwargs,
        )

        # Action feed forward
        act_attn_output = gate_msa_act * act_attn_output
        action_hidden_states = action_hidden_states + act_attn_output

        norm_action_hidden_states = self.norm2_act(action_hidden_states)
        norm_action_hidden_states = norm_action_hidden_states * (1 + scale_mlp_act) + shift_mlp_act
        ff_action_output = self.ff_act(norm_action_hidden_states)

        ff_action_output = gate_mlp_act * ff_action_output
        action_hidden_states = action_hidden_states + ff_action_output

        # attention outputs for the `obs_hidden_states`.
        if self.context_pre_only:
            # encoder_hidden_states = None
            obs_hidden_states = None # type:ignore
        else:
            assert gate_msa_obs is not None and \
                shift_mlp_obs is not None and \
                    scale_mlp_obs is not None and \
                        gate_mlp_obs is not None
            """ # lang forward
            lang_attn_output = c_gate_msa.unsqueeze(1) * lang_attn_output
            lang_hidden_states = lang_hidden_states + lang_attn_output
            norm_lang_hidden_states = self.norm2_lang(lang_hidden_states)
            norm_lang_hidden_states = norm_lang_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_lang, norm_lang_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_lang(norm_lang_hidden_states)
            lang_hidden_states = lang_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output """
            
            # obs feedforward
            assert self.norm2_obs is not None

            obs_attn_output = gate_msa_obs * obs_attn_output
            obs_hidden_states = obs_hidden_states + obs_attn_output

            norm_obs_hidden_states = self.norm2_obs(obs_hidden_states)
            norm_obs_hidden_states = norm_obs_hidden_states * (1 + scale_mlp_obs) + shift_mlp_obs

            assert self.ff_obs is not None
            ff_obs_output = self.ff_obs(norm_obs_hidden_states)
            ff_obs_output = gate_mlp_obs * ff_obs_output
            obs_hidden_states = obs_hidden_states + ff_obs_output

        return action_hidden_states, obs_hidden_states
    
class ActionTransformerModel(
    ModelMixin, ConfigMixin #, VLTTransformerActionMixin
    # PeftAdapterMixin, FromOriginalModelMixin, ReprodMixin, SD3Transformer2DLoadersMixin, 
):  
    """  An action transformer model """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ActionTransformerBlock"]
    # _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @classmethod
    def from_pretrained(cls,  pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # model = cls.from_config(config=None, **kwargs)
        model = ActionTransformerModel()
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        if pretrained_model_name_or_path is not None and "checkpoints" in str(pretrained_model_name_or_path):
            if os.path.exists(pretrained_model_name_or_path):
                 # NOTE use accelerator.checkpointing 
                from safetensors.torch import load_file
                file_path = f"{pretrained_model_name_or_path}/model.safetensors"
                assert os.path.exists(file_path), f"Checkpoint {file_path} does not exist."
                model_state_dict = load_file(file_path)
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                
                """ # load all checkpoint data, including weights, optimizer, etc.
                load_dict = torch.load(pretrained_model_name_or_path, weights_only=False) 
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = model.module if isinstance(model, DDP) else model
                missing_keys, unexpected_keys = model.load_state_dict(load_dict["model"], strict=False)  """

                if len(unexpected_keys) > 0:
                    logger.info(f"Unexpected keys in the checkpoint: {unexpected_keys}. ")
                if len(missing_keys) > 0:
                    logger.info(f"Missing keys in the checkpoint: {missing_keys}.")

                logger.info(f"Loaded model from {pretrained_model_name_or_path}")
            else:
                raise ValueError(f"Checkpoint {pretrained_model_name_or_path} does not exist. ")
        
        model.to(torch_dtype)
        return model

    @register_to_config
    def __init__(
        self,
        # sample_size: int = 128,
        # patch_size: int = 2,
        # in_channels: int = 16,
        # num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        # joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        # out_channels: int = 16,
        pos_embed_max_size: int = 192,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
        # VLA parameters
        training_phase: str = "joint",  # "joint", "action", "traj2d", "action_mini"
        # action head parameters
        action_num_blocks:int = 6,
        action_pred_horizon: int = 6,
        action_dim: int = 7,
        action_hidden_dim: int = 1536,
        action_nheads: int = 24,
        # action_attention_head_dim: Optional[int] = 64,
        n_conditions: int = 0,
        token_fusion: str = "concat",  # "concat", "cross"
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False,
        combined_temb: bool = False,
        final_layer_norm: bool = True,
        layerwise_vlm_fusion: bool = False,
        state_drop_prob: float = 0.0,
    ):
        super().__init__()
        # self.inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.state_drop_prob = state_drop_prob
        self.combined_temb = combined_temb
        if self.combined_temb:
            self.time_ins_embed = CombinedTimestepTextProjEmbeddingsND(
                embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
            ) # used; ND variant also accepts per-token (B,Tp) timesteps for RTC
        else:
            self.time_ins_embed = _TimeNetwork(time_dim=256, out_dim=action_hidden_dim)
        """ self.lang_embedder = nn.Linear(joint_attention_dim, caption_projection_dim) """

        # Set the observation projection
        self.obs_proj = ObservationProjection(
            action_pred_horizon=action_pred_horizon, 
            action_dim=action_dim, 
            hidden_dim=self.inner_dim, # not used
            output_dim=action_hidden_dim,
            n_conditions=n_conditions,
            token_fusion=token_fusion,
            resnet_store_path=resnet_store_path,
            odim=odim,
            view_feature_dim=view_feature_dim,
            use_film=use_film
        )

        total_params = sum(p.numel() for p in self.obs_proj.parameters() if p.requires_grad)
        logger.debug(f"ObservationEncoder parameters: {total_params:,}")
        # Set the action projection
        self.action_proj_in = ActionProjectionIn(
            action_pred_horizon=action_pred_horizon, 
            action_dim=action_dim, 
            output_dim=action_hidden_dim,
        )
        total_params = sum(p.numel() for p in self.action_proj_in.parameters() if p.requires_grad)
        logger.debug(f"ActionProjectionIn parameters: {total_params:,}")

        self.transformer_blocks = nn.ModuleList(
            [
                # IdentityBlock()
                VLATransformerBlock( # JointTransformerBlock
                    dim=action_hidden_dim,
                    num_attention_heads=action_nheads,
                    attention_head_dim=attention_head_dim,
                    # layerwise VLM fusion feeds each block a fresh VLM layer and discards the obs
                    # output, so make every block context_pre_only (obs = KV-only): no obs
                    # FF / to_add_out. Otherwise only the last block is context_pre_only.
                    context_pre_only=(layerwise_vlm_fusion or i == action_num_blocks - 1),
                    qk_norm=qk_norm,
                    # use_dual_attention=True if i in dual_attention_layers else False,
                    training_phase=training_phase,  # "joint", "action", "traj2d", "action_mini"
                )
                for i in range(action_num_blocks)
            ]
        )
        total_params = sum(p.numel() for p in self.transformer_blocks[0].parameters() if p.requires_grad)
        logger.debug(f"ActionTransformerBlock parameters: {total_params:,}")

        self.action_proj_out = ActionProjectionOut(
            hidden_size=action_hidden_dim,
            action_dim=action_dim,
            final_layer_norm=final_layer_norm
        )
        total_params = sum(p.numel() for p in self.action_proj_out.parameters())
        logger.debug(f"ActionProjectionOut parameters: {total_params:,}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], ActionTransformerModelOutput]: 
        assert joint_attention_kwargs is not None

        if self.config.n_conditions > 0: # type: ignore
            assert joint_attention_kwargs["traj2ds"].shape[1] == self.config.n_conditions, "model n_conditions does not match" # type: ignore

        # L
        if self.combined_temb:
            temb = self.time_ins_embed(timestep, pooled_projections)
        else: 
            temb = self.time_ins_embed(timestep)

        """ lang_hidden_states = self.lang_embedder(encoder_hidden_states) """

        # A
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")
        action_hidden_states = self.action_proj_in(noisy_action)

        # Obs: V+Proprio. views is (B, V, N, D): V==1 -> single last-layer context that
        # evolves through the blocks (default); V==n_blocks -> layerwise VLM fusion, block i gets
        # its own VLM layer as a FRESH context (tokenized per layer, obs not carried).
        views = joint_attention_kwargs["views"]

        # Random state dropout
        obs = joint_attention_kwargs["obs"]
        if self.training and self.state_drop_prob > 0.0:
            keep = (torch.rand(obs.shape[0], device=obs.device) >= self.state_drop_prob)
            obs = obs * keep.view(obs.shape[0], *([1] * (obs.dim() - 1))).to(obs.dtype)

        layerwise_vlm_fusion = views.shape[1] > 1
        if layerwise_vlm_fusion:
            assert views.shape[1] == len(self.transformer_blocks), (
                f"layerwise VLM fusion views has {views.shape[1]} layers != {len(self.transformer_blocks)} blocks"
            )
            per_block_obs = []
            obs_token_mask = None
            for i in range(views.shape[1]):
                obs_i, obs_token_mask = self.obs_proj(
                    views=views[:, i:i + 1],
                    obs=obs,
                    traj2ds=joint_attention_kwargs["traj2ds"],
                    text_embeddings=pooled_projections,
                    vlm_attn_mask=vlm_attn_mask,
                )
                per_block_obs.append(obs_i)
            obs_hidden_states = per_block_obs[0]
        else:
            obs_hidden_states, obs_token_mask = self.obs_proj(
                views=views,
                obs=obs,
                traj2ds=joint_attention_kwargs["traj2ds"],
                text_embeddings=pooled_projections,
                vlm_attn_mask=vlm_attn_mask
            ) # S, B, d_model

        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = True if skip_layers is not None and index_block in skip_layers else False
            if is_skip:
                continue

            # layerwise VLM fusion: feed this block its own VLM layer's fresh context and discard
            # the bidirectionally-updated obs (each block re-anchors to its VLM layer).
            obs_in = per_block_obs[index_block] if layerwise_vlm_fusion else obs_hidden_states
            action_hidden_states, obs_out = block(
                # hidden_states=hidden_states, # V
                # lang_hidden_states=lang_hidden_states, # L
                action_hidden_states=action_hidden_states, # A
                obs_hidden_states=obs_in, # O
                temb=temb,
                obs_token_mask=obs_token_mask
                # time_enc=time_enc,
                # obs_pos_emb=obs_pos_emb,
                # joint_attention_kwargs=joint_attention_kwargs,
            )
            if not layerwise_vlm_fusion:
                obs_hidden_states = obs_out

        action_output = self.action_proj_out(
            x=action_hidden_states,
            t=temb,
            # cond=obs_hidden_states,
        )
        return ActionTransformerModelOutput(action=action_output)


class DiTActionTransformerModel(
    ModelMixin, ConfigMixin #, VLTTransformerActionMixin
    # PeftAdapterMixin, FromOriginalModelMixin, ReprodMixin, SD3Transformer2DLoadersMixin, 
):  
    """  An action transformer model """

    _supports_gradient_checkpointing = True
    # _no_split_modules = ["ActionTransformerBlock"]
    # _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        # sample_size: int = 128,
        # patch_size: int = 2,
        # in_channels: int = 16,
        # num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        # joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        # out_channels: int = 16,
        pos_embed_max_size: int = 192,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
        # VLA parameters
        training_phase: str = "joint",  # "joint", "action", "traj2d", "action_mini"
        # action head parameters
        action_num_blocks:int = 6,
        action_pred_horizon: int = 6,
        action_dim: int = 7,
        action_hidden_dim: int = 1536, #this
        action_nheads: int = 24,
        # action_attention_head_dim: Optional[int] = 64,
        n_conditions: int = 0,
        token_fusion: str = "concat",  # "concat", "cross"
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False,
        combined_temb: bool = False
    ):
        super().__init__()
        # self.inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        # Set the observation projection
        self.obs_proj = ObservationProjection(
            action_pred_horizon=action_pred_horizon, 
            action_dim=action_dim, 
            hidden_dim=self.inner_dim, # not used
            output_dim=action_hidden_dim,
            n_conditions=n_conditions,
            token_fusion="concat",
            resnet_store_path=resnet_store_path,
            odim=odim,
            view_feature_dim=view_feature_dim,
            use_film=use_film
        )

        total_params = sum(p.numel() for p in self.obs_proj.parameters() if p.requires_grad)
        logger.debug(f"ObservationEncoder parameters: {total_params:,}")


        # q_former:
        # TODO: boqian: hardcode here
        num_query_tokens = 256
        num_heads_ = 16
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, action_hidden_dim))
        self.cross_attention = CrossAttentionBlock(action_hidden_dim, num_heads_)


        # assert action_hidden_dim == 1024
        self.DiT = DiT(
            in_channels=action_dim,
            depth=24,
            num_heads=16,
            token_size=action_hidden_dim,
            learn_sigma=False,
            class_dropout_prob=0,
            future_action_window_size=action_pred_horizon-1,
            past_action_window_size=0,
            n_conditon_token=num_query_tokens
        )
        total_params = sum(p.numel() for p in self.DiT.parameters() if p.requires_grad)
        logger.debug(f"DiT parameters: {total_params:,}")


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], ActionTransformerModelOutput]: 
        assert joint_attention_kwargs is not None

        if self.config.n_conditions > 0: # type: ignore
            assert joint_attention_kwargs["traj2ds"].shape[1] == self.config.n_conditions, "model n_conditions does not match" # type: ignore


        """ lang_hidden_states = self.lang_embedder(encoder_hidden_states) """

        # A
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")

        # Obs: V+Proprio
        condition_tokens, vlm_attn_mask = self.obs_proj(
            views=joint_attention_kwargs["views"],
            obs=joint_attention_kwargs["obs"],
            traj2ds=joint_attention_kwargs["traj2ds"],
            text_embeddings=pooled_projections,
            vlm_attn_mask=vlm_attn_mask
        ) # B, S, d_model

        # print("condition_tokens.shape:", condition_tokens.shape) # [B, 106, 1024])


        # q_former
        # TODO: forward vlm_attn_mask here 
        query = self.query_tokens.unsqueeze(0).expand(condition_tokens.shape[0], -1, -1) # (B, Q, D)
        queried = self.cross_attention(query, condition_tokens) # (B, Q, D)


        # DiT
        action_output = self.DiT(noisy_action, timestep, queried)
        

        return ActionTransformerModelOutput(action=action_output)


class HumanFoundationModel(nn.Module):

    def __init__(self, model_cfg, feature_extractors: dict[str, str]):
        super().__init__()
        
        # self.qwen = _QWen_VL_Interface(config=model_cfg)
        # self.layer_qformer = get_layerwise_qformer(config=self.config)
        # self.action_model = get_action_model(config=self.config)

        self.action_header = ActionTransformerModel(
            resnet_store_path=model_cfg.resnet_store_path,
            odim=model_cfg.odim,
            action_dim=model_cfg.action_dim,
            action_pred_horizon=model_cfg.action_chunk_size,
            view_feature_dim=model_cfg.view_feature_dim,
            use_film=model_cfg.use_film,
            combined_temb=model_cfg.combined_temb,
            action_hidden_dim=model_cfg.hidden_dim,
            )

        total_params = sum(p.numel() for p in self.action_header.parameters())
        overwatch.info(f"Total ActionTransformerModel parameters: {total_params:,}")

        vision_backbones = {}
        for backbone, model_path in feature_extractors.items():
            if backbone == "siglip":
                siglip = SiglipModel.from_pretrained(
                    model_path
                )  # , resume_download=None, local_files_only=True
                vision_backbones[backbone] = siglip.vision_model
                vision_backbones[backbone].head = nn.Identity() # remove classification head
                del siglip
            elif backbone == "dinov2":
                vision_backbones[backbone] = Dinov2Model.from_pretrained(
                    model_path
                )  # , resume_download=None, local_files_only=True
                vision_backbones[backbone].embeddings.mask_token = None
            else:
                raise ValueError(f"Unsupported vision backbone: {backbone}")
            
        self.vision_backbones = nn.ModuleDict(vision_backbones)

        for name, backbone in self.vision_backbones.items():
            num_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            overwatch.info(f"Vision backbone {name} has {num_parameters*1e-6:,}M parameters")
            
    def tune_vision_backbones(self, backbone_names: List[str]):
        for name, backbone in self.vision_backbones.items():
            if name in backbone_names:
                overwatch.info(f"Tuning vision backbone: {name}")
                for param in backbone.parameters():
                    param.requires_grad = True
            else:
                overwatch.info(f"Freezing vision backbone: {name}")
                for param in backbone.parameters():
                    param.requires_grad = False
    
    def extract_vision_features(self, batch):
        features = []
        for vision_backbone, model in self.vision_backbones.items():
            assert vision_backbone in batch
            B, V, C, H, W = batch[vision_backbone].shape
            if vision_backbone == "siglip":
                vision_outputs = model(
                    pixel_values=batch[vision_backbone].reshape(B * V, C, H, W),
                    return_dict=True,
                )
                siglip_features = vision_outputs.last_hidden_state
                features.append(siglip_features.view(B, V, *siglip_features.shape[1:]))
            elif vision_backbone == "dinov2":
                dinov2_outputs = model(
                    pixel_values=batch[vision_backbone].view(B * V, C, H, W)
                )
                dinov2_features = dinov2_outputs.last_hidden_state[:, 1:, :]
                features.append(dinov2_features.view(B, V, *dinov2_features.shape[1:]))
            else:
                raise ValueError(f"Unsupported vision backbone: {vision_backbone}")
        return torch.cat(features, dim=3)  # (B,V,N,D)
    
    def forward(
        self,
        action_samples: torch.Tensor,
        views: dict[str, torch.Tensor],
        states: torch.Tensor,
        # hidden_states: torch.Tensor,
        lang_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.LongTensor,
        # block_controlnet_hidden_states: List,
        # joint_attention_kwargs: Optional[Dict[str, Any]],
        traj2ds: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], HumanFoundationModelOutput]: 

        view_features = self.extract_vision_features(views)

        model_output = self.action_header(
            hidden_states=None,
            encoder_hidden_states=lang_hidden_states, # (B, 333, 4096)
            timestep=timestep,
            pooled_projections=pooled_projections, # (B, 2048)
            joint_attention_kwargs=dict(
                action_hidden_embeds=action_samples, # (B,Tp,Da)
                views=view_features,  # (B,V,N,D)
                obs=states,  # (B,1,M)
                traj2ds=traj2ds,  # (B, C, 3, H, W)
                # text_embeddings=text_embeddings
            ),
            return_dict=return_dict,
        )
        return HumanFoundationModelOutput(action=model_output.action)

    @torch.inference_mode()
    def predict_action(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: List[Any],
        instructions: List[str],
        **kwargs: str
    ) -> np.ndarray:
        # with torch.autocast("cuda", dtype=torch.float32):
        ...

    @torch.inference_mode()
    def chat(self):
        ...
    


class Psi0Model(nn.Module):

    def __init__(self, model_cfg, vlm_model: Qwen3VLForConditionalGeneration):
        super().__init__()
        
        if model_cfg.use_dit:
            self.action_header = DiTActionTransformerModel(
                resnet_store_path=model_cfg.resnet_store_path,
                odim=model_cfg.odim,
                action_dim=model_cfg.action_dim,
                action_pred_horizon=model_cfg.action_chunk_size,
                view_feature_dim=model_cfg.view_feature_dim,
                use_film=model_cfg.use_film,
                combined_temb=model_cfg.combined_temb,
                action_hidden_dim=model_cfg.hidden_dim,
                action_num_blocks=model_cfg.num_blocks,
                pooled_projection_dim=model_cfg.pooled_projection_dim,
            )
        else:
            self.action_header = ActionTransformerModel(
                resnet_store_path=model_cfg.resnet_store_path,
                odim=model_cfg.odim,
                action_dim=model_cfg.action_dim,
                action_pred_horizon=model_cfg.action_chunk_size,
                view_feature_dim=model_cfg.view_feature_dim,
                use_film=model_cfg.use_film,
                combined_temb=model_cfg.combined_temb,
                action_hidden_dim=model_cfg.hidden_dim,
                action_num_blocks=model_cfg.num_blocks,
                final_layer_norm=model_cfg.final_layer_norm,
                qk_norm=model_cfg.qk_norm,
                pooled_projection_dim=model_cfg.pooled_projection_dim,
                layerwise_vlm_fusion=model_cfg.vlm_layer_indices is not None,
                state_drop_prob=model_cfg.state_drop_prob,
            )

        total_params = sum(p.numel() for p in self.action_header.parameters())
        overwatch.info(f"Total ActionTransformerModel parameters: {total_params:,}")

        self.vlm_model = vlm_model
        total_params = sum(p.numel() for p in self.vlm_model.parameters())
        overwatch.info(f"Total VLM Backbone parameters: {total_params:,}")

        # Depth-matched conditioning: one VLM layer per action block. None -> last layer only.
        self.vlm_layer_indices = getattr(model_cfg, "vlm_layer_indices", None)
        if self.vlm_layer_indices is not None:
            assert not model_cfg.use_dit, "layerwise VLM fusion (vlm_layer_indices) is only supported for ActionTransformerModel (use_dit=False)"
            n_blocks = len(self.action_header.transformer_blocks)
            assert len(self.vlm_layer_indices) == n_blocks, (
                f"vlm_layer_indices has {len(self.vlm_layer_indices)} entries but the action "
                f"header has {n_blocks} blocks; they must match (one VLM layer per block)"
            )
            overwatch.info(f"Depth-matched VLM conditioning: block->layer map {self.vlm_layer_indices}")

    # load pretrained vlm+action head, and all the modules needed for predict_action
    @classmethod
    def from_pretrained(cls, run_dir, ckpt_step, launch_config:LaunchConfig, device):
        if not os.path.exists(run_dir):
            raise ValueError(f"Pretrained model path {run_dir} does not exist.")

        from safetensors.torch import load_file
        file_path = f"{run_dir}/checkpoints/ckpt_{ckpt_step}/model.safetensors"
        if not os.path.exists(file_path):
            raise ValueError(f"Checkpoint file {file_path} does not exist.")
        state_dict = load_file(file_path, device="cpu")

        # init empty vlm backbone from config only (skip loading base pretrained weights)
        vlm_config = AutoConfig.from_pretrained(QWEN3VL_VARIANT)
        vlm_config._attn_implementation = qwen3vl_attn_implementation()
        vlm_config.dtype = torch.bfloat16
        vlm_config.vision_config.dtype = torch.bfloat16
        vlm_config.text_config.dtype = torch.bfloat16
        vlm_model = Qwen3VLForConditionalGeneration(vlm_config)
        vlm_model = vlm_model.to(dtype=torch.bfloat16) # type: ignore

        vlm_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("action_header."):
                continue
            elif k.startswith('vlm_model.'):
                vlm_state_dict[k.replace("vlm_model.", "")] = v
            else:
                assert False, "check here"
        vlm_state_dict["lm_head.weight"] = vlm_state_dict["model.language_model.embed_tokens.weight"] # TODO check here

        if vlm_state_dict["lm_head.weight"].shape[0] != vlm_model.lm_head.weight.shape[0]:
            vlm_model.resize_token_embeddings(
                vlm_state_dict["lm_head.weight"].shape[0], 
                pad_to_multiple_of = 192,
                mean_resizing = True
            )
            overwatch.info(f"Resized model token embeddings to {vlm_model.lm_head.weight.shape[0]}")

        vlm_model.load_state_dict(vlm_state_dict, strict=True)
        overwatch.info("loaded vlm_backbone checkpoint successfully.")

        # init hfm-together model with vlm backbone
        model = Psi0Model(
            model_cfg=launch_config.model,
            vlm_model = vlm_model,
        )

        # load action head
        action_head_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vlm_model."):
                continue
            elif k.startswith('action_header.'):
                action_head_state_dict[k.replace("action_header.", "")] = v
            else:
                assert False, "check here"
        model.action_header.load_state_dict(action_head_state_dict, strict=True)
        overwatch.info("loaded action head checkpoint successfully.")

        # load necessary modules
        model.vlm_processor = AutoProcessor.from_pretrained(QWEN3VL_VARIANT)
        # model.tokenizer = model.vlm_processor.tokenizer

        if launch_config.model.noise_scheduler == "ddpm":
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
                scheduler = DDIMScheduler(
                    num_train_timesteps=launch_config.model.train_diffusion_steps,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule="squaredcos_cap_v2",
                    clip_sample=True,   
                    set_alpha_to_one=True,
                    steps_offset=0,
                    prediction_type="epsilon",
                )
        elif launch_config.model.noise_scheduler == "flow":
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
            scheduler= FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=launch_config.model.train_diffusion_steps, # MUST be 1000 as per pretrained SD3
            )
        else:
            raise ValueError(f"Unsupported noise scheduler: {launch_config.model.noise_scheduler}")

        model.noise_scheduler = scheduler
        model.action_horizon = launch_config.model.action_chunk_size
        model.action_dim = launch_config.model.action_dim
        model.device = device
        return model
    
    def _select_vlm_views(self, all_hidden_states) -> torch.Tensor:
        """Stack the VLM hidden states that condition the action header.

        Default (vlm_layer_indices is None): just the last layer -> (B, 1, N, D), fed to
        every action block (current behaviour). Depth-matched: one VLM layer per action
        block -> (B, n_blocks, N, D), so block i attends to VLM layer vlm_layer_indices[i]
        (openpi-style progressive features). `all_hidden_states` is the VLM's
        output_hidden_states tuple (index 0 = embeddings, 1..L = layers, -1 = last).
        """
        if self.vlm_layer_indices is None:
            return all_hidden_states[-1].unsqueeze(1)
        return torch.stack([all_hidden_states[i] for i in self.vlm_layer_indices], dim=1)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        action_samples: torch.Tensor,
        states: torch.Tensor,
        timestep: torch.LongTensor,
        traj2ds: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], HumanFoundationModelOutput]:

        # extract vision + language features (keep all layers for layerwise VLM fusion conditioning)
        all_hidden_states = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        ).hidden_states
        vlm_hidden_states = all_hidden_states[-1]     # last layer, for logging (norm)
        views = self._select_vlm_views(all_hidden_states)  # (B, V, N, D): V=1 or n_blocks

        # Mean per-token L2 norm of the VLM feature right before it is handed to the
        # action header, for logging/monitoring. Detached + fp32; masked to valid tokens
        # (pads carry non-zero norm) when the attention mask lines up.
        with torch.no_grad():
            tok_norm = vlm_hidden_states.detach().float().norm(dim=-1)  # (B, N)
            if attention_mask is not None and attention_mask.shape == tok_norm.shape:
                m = attention_mask.to(tok_norm.dtype)
                vlm_feat_norm = (tok_norm * m).sum() / m.sum().clamp_min(1.0)
            else:
                vlm_feat_norm = tok_norm.mean()

        model_output = self.action_header(
            hidden_states=None,
            timestep=timestep,
            pooled_projections=pooled_projections,  # (B, pooled_projection_dim), for combined_temb
            joint_attention_kwargs=dict(
                action_hidden_embeds=action_samples, # (B,Tp,Da)
                views=views,  # (B,V,N,D)  V=1 (last layer) or n_blocks (layerwise VLM fusion)
                obs=states,  # (B,1,M)
                traj2ds=traj2ds,  # (B, C, 3, H, W)
            ),
            vlm_attn_mask=attention_mask, # (B, seq_len)
            return_dict=return_dict,
        )
        return HumanFoundationModelOutput(action=model_output.action, vlm_feat_norm=vlm_feat_norm)

    def _collate_vlm_batch(self, batch_input_ids: List[torch.Tensor]):
        """Right-pad mixed-length samples like training's PaddedCollatorForTogether
        (trainers/qwen3vl_mixin.py); the mask must also reach the action header as
        `vlm_attn_mask` or it attends over the pads. Padding shifts the obs token's position
        index, so a batched sample is not bit-identical to a solo one (measured harmless).
        """
        pad_id = self.vlm_processor.tokenizer.pad_token_id
        assert pad_id is not None, "tokenizer has no pad_token_id; cannot pad a mixed batch"
        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_id)
        return input_ids, input_ids.ne(pad_id)

    @torch.inference_mode()
    def predict_action(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds,
        goal_images: List[List[Image.Image]] | None = None,  # B * List of PIL goal images
        pooled_projections: Optional[torch.Tensor] = None,  # (B, D) for combined_temb
        **kwargs: str
    ) -> torch.Tensor:

        bsz = states.shape[0]
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []


        for b, (observation, instruction) in enumerate(zip(observations, instructions)):
            messages = []
            content = [{"type": "image", "image": img} for img in observation]
            content.append({"type": "text", "text": instruction})
            # Match training (PsixModelTransform.build_qwenvl_inputs): goal image(s)
            # are appended AFTER the instruction text.
            if goal_images is not None and goal_images[b] is not None:
                content.extend({"type": "image", "image": img} for img in goal_images[b])
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = self.vlm_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            # input_ids torch.Size([1, 80])
            # attention_mask torch.Size([1, 80])
            # pixel_values torch.Size([256, 1536])
            # image_grid_thw torch.Size([1, 3])
            batch_input_ids.append(inputs['input_ids'].squeeze(0))
            batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
            batch_pixel_values.append(inputs['pixel_values'])
            batch_image_grid_thw.append(inputs['image_grid_thw'])

        # Per-sample tokenization means different instructions -> different lengths, so stack()
        # would fail. Right-pad exactly like training's collator (see _collate_vlm_batch).
        batch_input_ids, batch_attention_mask = self._collate_vlm_batch(batch_input_ids)
        # Qwen expects pixel_values / image_grid_thw flattened across ALL images in the
        # batch (no leading batch dim). With >1 image per sample (ego + goal) `stack`
        # would make image_grid_thw 3-D -> thw[:, 2] crashes. `cat` keeps them 2-D:
        #   pixel_values   -> (total_image_patches, 1536)
        #   image_grid_thw -> (total_images, 3)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            # extract vision + language features
            output = self.vlm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
            

            # use hidden states from the last layer
            vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
            vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

            # generate action from noise
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)

            for timestep in self.noise_scheduler.timesteps:
                batched_timestep = timestep.expand(bsz).to(self.device)
                model_pred = self.action_header(
                    hidden_states=None,
                    timestep=batched_timestep,
                    pooled_projections=pooled_projections,  # (B, D) for combined_temb
                    joint_attention_kwargs=dict(
                        action_hidden_embeds=action_samples, # (B,Tp,Da)
                        views=self._select_vlm_views(vlm_hidden_states_),  # (B,V,N,D)
                        obs=states,  # (B,1,M)
                        traj2ds=traj2ds,  # (B, C, 3, H, W)
                    ),
                    # match training: mask the pad positions out of the action header's
                    # cross-attention, otherwise it reads pad-token hidden states
                    vlm_attn_mask=batch_attention_mask,  # (B, seq_len)
                    return_dict=True,
                ).action
                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                ).prev_sample

        return action_samples.float()

    @torch.inference_mode()
    def predict_action_with_training_rtc_flow(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (B, H, D) — one previous chunk per sample
        inference_delay: int = 0,
        max_delay: int = 0,
        goal_images: List[List[Image.Image]] | None = None,  # B * List of PIL goal images
        pooled_projections: Optional[torch.Tensor] = None,  # (B, D) for combined_temb
        **kwargs: str
    ) -> torch.Tensor:

        ## RTC related ##
        H = self.action_horizon
        assert prev_actions is not None and inference_delay > 0 and max_delay > 0, "prev_actions, inference_delay and max_delay must be provided"
        # One previous chunk per sample. Deliberately NOT allowing a (1, H, D) broadcast at
        # bsz > 1: that would silently condition every client in a batch on one client's
        # previous chunk. Single-client callers have states.shape[0] == 1 and are unaffected.
        assert prev_actions.shape[0] == states.shape[0], (
            f"prev_actions batch {prev_actions.shape[0]} != states batch {states.shape[0]}")
        prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

        # Create soft mask for inpainting
        d = inference_delay

        # Validate constraint from paper: d ≤ s ≤ H - d
        assert d < H, f"Constraint violated: d={d}, H={H}. Need d < H"
        # assert d < max_delay, f"Constraint violated: d={d}, max_delay={max_delay}. Need d < max_delay"
        d_limit = min(H, max_delay)
        if d >= d_limit:
            overwatch.warning(
                f"RTC inference_delay d={d} >= limit {d_limit}=min(H={H}, max_delay={max_delay}); "
                f"clamping to {d_limit - 1} — inference is slower than the delay budget"
            )
            d = d_limit - 1

        bsz = states.shape[0]
        # (B, H): d is shared by all samples, but expand explicitly — `batched_timestep_masked`
        # below is derived from this mask and must carry the batch dim, not rely on broadcasting.
        prefix_mask = (torch.arange(H, device=self.device)[None, :] < d).expand(bsz, -1)

        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []


        for b, (observation, instruction) in enumerate(zip(observations, instructions)):
            messages = []
            content = [{"type": "image", "image": img} for img in observation]
            content.append({"type": "text", "text": instruction})
            # Match training (PsixModelTransform.build_qwenvl_inputs): goal image(s)
            # are appended AFTER the instruction text.
            if goal_images is not None and goal_images[b] is not None:
                content.extend({"type": "image", "image": img} for img in goal_images[b])
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = self.vlm_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            # input_ids torch.Size([1, 80])
            # attention_mask torch.Size([1, 80])
            # pixel_values torch.Size([256, 1536])
            # image_grid_thw torch.Size([1, 3])
            batch_input_ids.append(inputs['input_ids'].squeeze(0))
            batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
            batch_pixel_values.append(inputs['pixel_values'])
            batch_image_grid_thw.append(inputs['image_grid_thw'])

        # Per-sample tokenization means different instructions -> different lengths, so stack()
        # would fail. Right-pad exactly like training's collator (see _collate_vlm_batch).
        batch_input_ids, batch_attention_mask = self._collate_vlm_batch(batch_input_ids)
        # Qwen expects pixel_values / image_grid_thw flattened across ALL images in the
        # batch (no leading batch dim). With >1 image per sample (ego + goal) `stack`
        # would make image_grid_thw 3-D -> thw[:, 2] crashes. `cat` keeps them 2-D:
        #   pixel_values   -> (total_image_patches, 1536)
        #   image_grid_thw -> (total_images, 3)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            # extract vision + language features
            output = self.vlm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
            

            # use hidden states from the last layer
            vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
            vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

            # generate action from noise
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)

            # self.noise_scheduler.timesteps: tensor([1000.,  889.,  778.,  667.,  556.,  445.,  334.,  223.,  112.,    1.])
            # self.noise_scheduler.sigmas: tensor([1.0000, 0.8890, 0.7780, 0.6670, 0.5560, 0.4450, 0.3340, 0.2230, 0.1120, 0.0010, 0.0000])

            for i, timestep in enumerate(self.noise_scheduler.timesteps):
                # batched_timestep = timestep.expand(bsz).to(self.device).detach()

                batched_timestep_masked = torch.where(prefix_mask, 0, timestep.to(self.device)) # shape (B, H)

                # replace action_samples with clean prev_actions when prefix_mask == True
                action_samples = torch.where(prefix_mask[:, :, None], prev_actions, action_samples)

                model_pred = self.action_header(
                    hidden_states=None,
                    timestep=batched_timestep_masked,
                    pooled_projections=pooled_projections,  # (B, D) for combined_temb
                    joint_attention_kwargs=dict(
                        action_hidden_embeds=action_samples, # (B,Tp,Da)
                        views=self._select_vlm_views(vlm_hidden_states_),  # (B,V,N,D)
                        obs=states,  # (B,1,M)
                        traj2ds=traj2ds,  # (B, C, 3, H, W)
                    ),
                    # match training: mask the pad positions out of the action header's
                    # cross-attention, otherwise it reads pad-token hidden states
                    vlm_attn_mask=batch_attention_mask,  # (B, seq_len)
                    return_dict=True,
                ).action

                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                ).prev_sample

                # if i == len(self.noise_scheduler.timesteps) - 1:
                #     action_samples = torch.where(prefix_mask[:, :, None], prev_actions, action_samples)

        return action_samples.float()

    @torch.no_grad()
    def predict_action_with_rtc_flow(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (1, H, D)
        inference_delay: int = 0,
        # s: free steps at the tail. None = derive H-2d. Matches
        # psix.predict_action_with_test_time_rtc -- and must stay `is not None`
        # below, so that an explicit s=0 is honoured rather than silently derived.
        execution_horizon: int | None = None,
        pooled_projections: Optional[torch.Tensor] = None,  # (B, D) for combined_temb
        mask_schedule: str = "exponential",
        guidance_weight: float = 5.0,  # DEPRECATED, see the guidance scaling note below
        guidance_alpha: float = 0.9,
        **kwargs: str
    ) -> np.ndarray:

        ## RTC related ##
        H = self.action_horizon
        assert prev_actions is not None, "prev_actions must be provided"
        assert prev_actions.shape[0] == 1
        assert tuple(prev_actions.shape) == (1, H, self.action_dim), (
            f"prev_actions shape {tuple(prev_actions.shape)} != (1,{H},{self.action_dim}); "
            "the caller must shift/pad the previous chunk to length H")
        prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

        # Create soft mask for inpainting
        d = int(inference_delay)
        s = int(execution_horizon) if execution_horizon is not None else max(1, H - 2 * d)

        # Paper constraint d ≤ s ≤ H-d. Clamp instead of asserting (matches
        # psix.predict_action_with_test_time_rtc): a server that hands over a
        # d/s pair one tick outside the window should degrade to the nearest
        # valid mask, not kill the request mid-episode. d=0 is legal here -- it
        # just means nothing is hard-frozen; rows [0, H-s) still carry the ramp,
        # and only s == H reduces this call to plain predict_action.
        d = max(0, min(d, H - 1))
        s = max(d, min(s, H - d))

        # Create soft mask [H] according to Equation 5
        mask = self._create_soft_mask(
            H, d, s, schedule=mask_schedule, device=self.device
        )
        mask_expanded = mask.view(1, -1, 1).expand(1, self.action_horizon, self.action_dim)

        bsz = states.shape[0]
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []

        for observation, instruction in zip(observations, instructions):
            messages = []
            content = [{"type": "image", "image": img} for img in observation]
            content.append({"type": "text", "text": instruction})
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = self.vlm_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            batch_input_ids.append(inputs['input_ids'].squeeze(0))
            batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
            batch_pixel_values.append(inputs['pixel_values'])
            batch_image_grid_thw.append(inputs['image_grid_thw'])

        # Same collation as predict_action / predict_action_with_training_rtc_flow.
        # This path used to torch.stack() everything, which gave pixel_values a
        # leading batch dim (B, patches, 1536) -- a shape Qwen does not accept --
        # and could not pad ragged prompts at all. Keep the three sampling paths
        # byte-identical up to the guidance term, or a test-time RTC rollout is
        # not comparable with the open-loop one.
        batch_input_ids, batch_attention_mask = self._collate_vlm_batch(batch_input_ids)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # extract vision + language features
            output = self.vlm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29

            # use hidden states from the last layer
            vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
            vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

            # generate action from noise
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            # target_noise = action_samples.clone().detach()
            self.noise_scheduler.set_timesteps(num_inference_steps)

            for timestep in self.noise_scheduler.timesteps:
                batched_timestep = timestep.expand(bsz).to(self.device).detach()

                with torch.enable_grad():
                    # Detach and enable gradient only for sample_actions in this step
                    action_samples_grad = action_samples.detach().requires_grad_(True)

                    model_pred = self.action_header(
                        hidden_states=None,
                        timestep=batched_timestep,
                        pooled_projections=pooled_projections,  # (B, D) for combined_temb
                        joint_attention_kwargs=dict(
                            action_hidden_embeds=action_samples_grad, # (B,Tp,Da)
                            views=self._select_vlm_views(vlm_hidden_states_),  # (B,V,N,D)
                            obs=states,  # (B,1,M)
                            traj2ds=traj2ds,  # (B, C, 3, H, W)
                        ),
                        # match training: mask the pad positions out of the action
                        # header's cross-attention, otherwise it reads pad-token
                        # hidden states
                        vlm_attn_mask=batch_attention_mask,  # (B, seq_len)
                        return_dict=True,
                    ).action

                    # 1. 外推干净data X_0
                    tau = self.noise_scheduler.sigmas[self.noise_scheduler.index_for_timestep(timestep)].detach()
                    pred_x0 = action_samples_grad - tau * model_pred
                    # 2. 计算(Y - pred_X_0) * weights
                    error = (prev_actions - pred_x0.detach()) * mask_expanded
                    # 3. vjp
                    pinv_correction = torch.autograd.grad(
                        outputs=pred_x0,  # [1,T,D]
                        inputs=action_samples_grad,        # [1,T,D]
                        grad_outputs=error,         # 匹配形状 [1,T,D]
                        retain_graph=False,
                        create_graph=False
                    )[0]  # 输出 [1,T,D]
                    # print("pinv_correction:", pinv_correction)

                # 4. 缩放修正项，修正v
                pinv_norm = pinv_correction.norm()
                if tau.item() >= 1.0 or pinv_norm < 1e-8:
                    # tau == 1: the sample is still pure noise, there is nothing to guide yet
                    correction = torch.zeros_like(pinv_correction)
                else:
                    scale = guidance_alpha * error.norm() / (tau * pinv_norm)
                    correction = scale * pinv_correction

                model_pred = model_pred.detach() - correction

                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                ).prev_sample

        return action_samples.float()

    def predict_action_with_rtc_flow_naive_inpaint(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (1, H, D)
        inference_delay: int = 0,
        execution_horizon: int = 0,
        mask_schedule: str = "exponential",
        guidance_weight: float = 5.0,
        **kwargs: str
    ) -> np.ndarray:
        with torch.no_grad():

            ## RTC related ##
            H = self.action_horizon
            assert prev_actions is not None and inference_delay > 0 and execution_horizon > 0, "prev_actions, inference_delay and execution_horizon must be provided"
            assert prev_actions.shape[0] == 1
            prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

            # Create soft mask for inpainting
            d = inference_delay
            s = execution_horizon

            # Validate constraint from paper: d ≤ s ≤ H - d
            assert d <= s and s <= H - d, f"Constraint violated: d={d}, s={s}, H={H}. Need d ≤ s ≤ H-d"

            # Create soft mask [H] according to Equation 5
            mask = self._create_soft_mask(
                H, d, s, schedule=mask_schedule, device=self.device
            )
            mask_expanded = mask.view(1, -1, 1).expand(1, self.action_horizon, self.action_dim)
            ##              ##


            bsz = states.shape[0]
            batch_input_ids = []
            batch_attention_mask = []
            batch_pixel_values = []
            batch_image_grid_thw = []


            for observation, instruction in zip(observations, instructions):
                messages = []
                content = [{"type": "image", "image": img} for img in observation]
                content.append({"type": "text", "text": instruction})
                user_msg = {"role": "user", "content": content}
                messages.append([user_msg])
                texts = [
                    self.vlm_processor.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                    for m in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
                inputs = self.vlm_processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                # input_ids torch.Size([1, 80])
                # attention_mask torch.Size([1, 80])
                # pixel_values torch.Size([256, 1536])
                # image_grid_thw torch.Size([1, 3])
                batch_input_ids.append(inputs['input_ids'].squeeze(0))
                batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
                batch_pixel_values.append(inputs['pixel_values'])
                batch_image_grid_thw.append(inputs['image_grid_thw'].squeeze(0))

            batch_input_ids = torch.stack(batch_input_ids) # (B, 80)
            batch_attention_mask = torch.stack(batch_attention_mask) # (B, 80)
            batch_pixel_values = torch.stack(batch_pixel_values) # (B, 256, 1536)
            batch_image_grid_thw = torch.stack(batch_image_grid_thw) # (B, 3)

            with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
                # extract vision + language features
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values=batch_pixel_values,
                    image_grid_thw=batch_image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True
                )
                vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
                

                # use hidden states from the last layer
                vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
                vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

                # generate action from noise
                action_samples = torch.randn(
                    bsz, self.action_horizon, self.action_dim, device=self.device
                )
                self.noise_scheduler.set_timesteps(num_inference_steps)

                # Fixed noise for consistency in inpainting
                # Use the same noise as action_samples to ensure the whole sequence starts from the same latent x_1
                target_noise = action_samples.clone()

                for i, timestep in enumerate(self.noise_scheduler.timesteps):
                    batched_timestep = timestep.expand(bsz).to(self.device).detach()

                    ### Naive inpainting (Repaint method):
                    # Step 1: Add noise to prev_actions matching current timestep's noise level
                    # This ensures consistency in noise levels across the entire action sequence
                    tau = self.noise_scheduler.sigmas[self.noise_scheduler.index_for_timestep(timestep)]
                    noisy_prev_actions = (1 - tau) * prev_actions + tau * target_noise
                    
                    # Step 2: Replace first d positions with noisy prev_actions (hard mask)
                    action_samples[:, :d, :] = noisy_prev_actions[:, :d, :]

                    # Step 3: Model prediction
                    model_pred = self.action_header(
                        hidden_states=None,
                        timestep=batched_timestep,
                        joint_attention_kwargs=dict(
                            action_hidden_embeds=action_samples, # (B,Tp,Da)
                            views=self._select_vlm_views(vlm_hidden_states_),  # (B,V,N,D)
                            obs=states,  # (B,1,M)
                            traj2ds=traj2ds,  # (B, C, 3, H, W)
                        ),
                        return_dict=True,
                    ).action

                    # Step 4: Denoise one step (updates all positions including first d)
                    action_samples = self.noise_scheduler.step(
                        model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                    ).prev_sample
                    
                    # Step 5: Re-inpaint with noise level matching the denoised state
                    # TODO: do this or not??? boqian
                    if i == len(self.noise_scheduler.timesteps) - 1:
                        # Last step: use clean prev_actions (tau = 0)
                        action_samples[:, :d, :] = prev_actions[:, :d, :]

            return action_samples

    def _create_soft_mask(self, H, d, s, schedule="exponential", device="cpu"):
        """
        Create soft mask for RTC inpainting (Equation 5 in paper).
        
        Paper: "Real-Time Execution of Action Chunking Flow Policies"
        Figure 3 and Equation 5:
        
        W_i = { 1                           if i < d (frozen region)
              { c_i * (e^(c_i) - 1)/(e - 1) if d ≤ i < H - s (intermediate region)
              { 0                           if i ≥ H - s (free region)
        
        where c_i = (H - s - i)/(H - s - d + 1)
        
        Args:
            H: Prediction horizon (total sequence length)
            d: Inference delay (number of frozen steps, already executed)
            s: Execution horizon (non-overlapping steps at end). 
            schedule: 'exponential' (paper default), 'linear', 'hard', or 'simple'
        
        Returns:
            mask: [H] tensor with values in [0, 1]
                  Guidance weights for each timestep
        """
        mask = torch.zeros(H, device=device)
        
        if schedule == "hard":
            # Hard mask: 1 for frozen, 0 for rest
            mask[:d] = 1.0
        
        elif schedule == "linear":
            # Linear decay
            mask[:d] = 1.0
            # Three-region version (full paper)
            overlap_end = H - s
            if d < overlap_end:
                # Linear decay from 1 to 0 in intermediate region
                indices = torch.arange(d, overlap_end, device=device).float()
                mask[d:overlap_end] = 1.0 - (indices - d) / (overlap_end - d)
            # mask[overlap_end:] remains 0
        
        elif schedule == "exponential":
            # Exponential decay (paper default, Equation 5)
            mask[:d] = 1.0
            
            # Three-region version (full paper implementation)
            overlap_end = H - s
            if d < overlap_end:
                indices = torch.arange(d, overlap_end, device=device).float()
                # c_i = (H - s - i) / (H - s - d + 1)
                c_i = (overlap_end - indices) / (overlap_end - d + 1)
                # W_i = c_i * (e^(c_i) - 1) / (e - 1)
                e = torch.tensor(torch.e, device=device)
                mask[d:overlap_end] = c_i * (torch.exp(c_i) - 1) / (e - 1)
            # mask[overlap_end:] remains 0
        
        elif schedule == "simple":
            # Simplified exponential (for backward compatibility)
            mask[:d] = 1.0
            if d < H:
                indices = torch.arange(d, H, device=device).float()
                mask[d:] = torch.exp(-5.0 * (indices - d) / (H - d))
        
        else:
            raise ValueError(f"Unknown mask schedule: {schedule}")
        
        return mask
    
    @torch.inference_mode()
    def chat(self):
        ...
    
