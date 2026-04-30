"""T5ForMultimodalGeneration — Gated Vision-Language Fusion on T5.

Architecture (MedThink paper):
  1. TextualEncoder: stock T5 encoder (UnifiedQA-T5-base)
  2. VisualEncoder:  pre-extracted DETR features [B, 100, 256]
  3. Cross-Attention: MHA(Q=text, K=image, V=image)
  4. Gated Fusion:   lambda=sigma(W*[F_T;H_attn]), F_fuse=(1-lambda)*F_T + lambda*H_attn
  5. TextualDecoder: stock T5 decoder generates answer + rationale
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    """T5 with gated cross-attention fusion of pre-extracted image features."""

    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        r"image_dense\.", r"image_norm\.",
        r"q_proj\.", r"k_proj\.", r"v_proj\.", r"out_proj\.",
        r"gate_dense\.",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config, patch_size=(100, 256)):
        super().__init__(config)

        patch_num, patch_dim = patch_size
        d = config.d_model  # 768 for t5-base

        # Image projection + normalization
        self.image_dense = nn.Linear(patch_dim, d)
        self.image_norm = nn.LayerNorm(d)

        # Manual cross-attention
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        # Gated fusion
        self.gate_dense = nn.Linear(2 * d, d)
        self.gate_act = nn.Sigmoid()

        # Explicit init with proper scaling (do NOT use post_init —
        # it applies T5's initializer_factor=1.0 as std, causing 10^38 overflow)
        for layer in [self.image_dense, self.q_proj, self.k_proj,
                      self.v_proj, self.out_proj, self.gate_dense]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _fuse_image_features(self, hidden_states, image_ids):
        """Gated cross-attention fusion: Q=text, K=V=image."""
        # Project and normalize image features
        image_emb = self.image_norm(self.image_dense(image_ids))  # [B,100,d]

        # Cross-attention with manual softmax (numerically safe)
        Q = self.q_proj(hidden_states)       # [B, seq, d]
        K = self.k_proj(image_emb)           # [B, 100, d]
        V = self.v_proj(image_emb)           # [B, 100, d]

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.clamp(-50, 50)       # prevent softmax overflow
        attn_weights = F.softmax(scores, dim=-1)
        image_att = self.out_proj(torch.matmul(attn_weights, V))  # [B, seq, d]

        # Gated fusion: lambda = sigma(W * [F_T ; H_attn])
        gate = self.gate_act(self.gate_dense(
            torch.cat([hidden_states, image_att], dim=-1)
        ))
        return (1 - gate) * hidden_states + gate * image_att

    def forward(
        self,
        input_ids=None, image_ids=None, attention_mask=None,
        decoder_input_ids=None, decoder_attention_mask=None,
        head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None,
        encoder_outputs=None, past_key_values=None,
        inputs_embeds=None, decoder_inputs_embeds=None,
        labels=None, use_cache=None,
        output_attentions=None, output_hidden_states=None,
        return_dict=None, **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Step 1: Encode text
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            # Step 2: Fuse with image
            if image_ids is not None:
                fused = self._fuse_image_features(
                    encoder_outputs.last_hidden_state, image_ids
                )
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=fused,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Step 3: Prepare decoder inputs
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # Step 4: Decode
        decoder_kwargs = dict(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values,
            encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
        if "cache_position" in kwargs:
            decoder_kwargs["cache_position"] = kwargs["cache_position"]

        decoder_outputs = self.decoder(**decoder_kwargs)
        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # Step 5: Loss
        loss = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss, logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        image_ids = kwargs.pop("image_ids", None)
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if image_ids is not None:
            model_inputs["image_ids"] = image_ids
        return model_inputs

    def test_step(self, tokenizer, batch, **kwargs):
        device = next(self.parameters()).device
        output = self.generate(
            input_ids=batch["input_ids"].to(device),
            image_ids=batch["image_ids"].to(device),
            **kwargs,
        )
        return {
            "preds": tokenizer.batch_decode(output, skip_special_tokens=True),
            "targets": tokenizer.batch_decode(batch["labels"], skip_special_tokens=True),
        }