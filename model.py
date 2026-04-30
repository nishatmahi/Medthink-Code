"""T5ForMultimodalGeneration — Gated Vision-Language Fusion on T5.

Architecture (MedThink paper):
  1. TextualEncoder: stock T5 encoder (UnifiedQA-T5-base)
  2. VisualEncoder:  pre-extracted DETR features [B, 100, 256]
  3. Cross-Attention: MHA(Q=text, K=image, V=image)
  4. Gated Fusion:   lambda=sigma(W*[F_T;H_attn]), F_fuse=(1-lambda)*F_T + lambda*H_attn
  5. TextualDecoder: stock T5 decoder generates answer + rationale

Design: image fusion runs AFTER the stock T5 encoder, not inside it.
This avoids reimplementing T5Stack internals and works with any
transformers version (v4.x and v5+).
"""

from typing import Optional, Tuple, Union

import torch
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
        # Multimodal layers — expected missing from pretrained T5 checkpoint
        r"image_dense\.",
        r"image_norm\.",
        r"mha_layer\.",
        r"gate_dense\.",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config, patch_size=(100, 256)):
        super().__init__(config)

        # ── Multimodal fusion layers (on top of T5, NOT inside T5Stack) ──
        patch_num, patch_dim = patch_size
        self.image_dense = nn.Linear(patch_dim, config.d_model)
        self.image_norm = nn.LayerNorm(config.d_model)
        self.mha_layer = nn.MultiheadAttention(
            embed_dim=config.d_model,
            kdim=config.d_model,
            vdim=config.d_model,
            num_heads=1,
            batch_first=True,
        )
        self.gate_dense = nn.Linear(2 * config.d_model, config.d_model)
        self.gate_act = nn.Sigmoid()

        # Initialize the new layers (pretrained T5 weights load after this)
        self.post_init()

    # ------------------------------------------------------------------ #
    #  Image fusion                                                       #
    # ------------------------------------------------------------------ #

    def _fuse_image_features(self, hidden_states, image_ids):
        """Gated cross-attention fusion of text and image features.

        Args:
            hidden_states: Encoder output [B, seq_len, d_model]
            image_ids: DETR features [B, num_patches, patch_dim]

        Returns:
            Fused hidden states [B, seq_len, d_model]
        """
        # Project image patches → d_model and normalize to prevent overflow
        image_emb = self.image_norm(self.image_dense(image_ids))

        # Cross-attention: Q=text features, K=V=image features
        image_att, _ = self.mha_layer(hidden_states, image_emb, image_emb)

        # Gated fusion: lambda = sigma(W · [F_T ; H_attn])
        merge = torch.cat([hidden_states, image_att], dim=-1)
        gate = self.gate_act(self.gate_dense(merge))

        # F_fuse = (1 - lambda) * F_T + lambda * H_attn
        return (1 - gate) * hidden_states + gate * image_att

    # ------------------------------------------------------------------ #
    #  Forward pass                                                       #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ── Step 1: Encode text using stock T5 encoder ──
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            # ── Step 2: Fuse with image features ──
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

        # ── Step 3: Prepare decoder inputs ──
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # ── Step 4: Decode using stock T5 decoder ──
        # Build decoder kwargs (only pass params the decoder accepts)
        decoder_kwargs = dict(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Forward cache_position if present (transformers v5+ generation)
        if "cache_position" in kwargs:
            decoder_kwargs["cache_position"] = kwargs["cache_position"]

        decoder_outputs = self.decoder(**decoder_kwargs)

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # ── Step 5: Compute loss ──
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # ------------------------------------------------------------------ #
    #  Generation support                                                 #
    # ------------------------------------------------------------------ #

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Extend parent's method to forward image_ids through generation."""
        image_ids = kwargs.pop("image_ids", None)
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if image_ids is not None:
            model_inputs["image_ids"] = image_ids
        return model_inputs

    def test_step(self, tokenizer, batch, **kwargs):
        """Run generation on a batch and return predicted + target strings."""
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device)
        image_ids = batch["image_ids"].to(device)

        output = self.generate(
            input_ids=input_ids,
            image_ids=image_ids,
            **kwargs,
        )

        return {
            "preds": tokenizer.batch_decode(output, skip_special_tokens=True),
            "targets": tokenizer.batch_decode(batch["labels"], skip_special_tokens=True),
        }