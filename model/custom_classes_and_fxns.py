# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 07:39:20 2022

@author: brand
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

EncodedInput = List[int]

# from transformers.utils import (
#     DUMMY_INPUTS,
#     DUMMY_MASK,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     is_torch_fx_proxy,
#     logging,
#     replace_return_docstrings,
# )

# Warning messafe for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

_CONFIG_FOR_DOC = "T5Config"



def remove_def_markers(example, def_span_indices):
    """
    Remove all definiendum span markers from the dataset.
    """
    begin,end = def_span_indices
    example['input_ids'].pop(end)
    example['input_ids'].pop(begin)
    example['attention_mask'].pop(end)
    example['attention_mask'].pop(begin)
    example['cross_attention_mask'].pop(end)
    example['cross_attention_mask'].pop(begin)
    
    return example


def prepare_for_xattn(example, tokenizer, demarcator):
    """
    Add cross-attention mask and remove temporary definiendum span markers
    from the data.
    """
    # Only definiendum span and eos_token will be unmasked for cross-attention
    def_ids = tokenizer.convert_tokens_to_ids([demarcator, tokenizer.eos_token])
    def_indices = []
    sent = example['input_ids']
    
    for i, token_id in enumerate(sent):
        if token_id in def_ids:
            def_indices.append(i)
            
    # assert len(def_indices) == 3, "Definiendum span not found. def_indices should consist of 3 integers but is instead " + str(def_indices) + " (Length: " + str(len(sent)) + ")\n" + tokenizer.decode(sent)
    if len(def_indices) == 3: # Definiendum span found (plus eos token).
        begin,end = def_indices[:2]
        eos_index = def_indices[-1]
        
    elif len(def_indices) == 1: # Definiendum span not found (just eos token).
        begin,end = [0,len(sent)-1]
        eos_index = def_indices[0]
    
    else:
        raise Exception("Did not find two definiendum markers.\n" + tokenizer.decode(sent))
    
    # Mask everything except for definiendum
    cross_attention_mask = [0]*len(sent)
    cross_attention_mask[begin:end] = [1]*(end-begin)
    cross_attention_mask[eos_index] = 1
    example['cross_attention_mask'] = cross_attention_mask
    
    # Remove definiendum markers
    if len(def_indices) == 3: # Definiendum markers found
        example = remove_def_markers(example, (begin,end))
    
    return example


class TokenizerWithXMask(T5Tokenizer):
    """
    This updates the T5Tokenizer to also perform dynamic padding of provided
    cross_attention_masks (inside the collater fxn).
    """

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            return_xattn_mask: Optional[bool] = None,
        ) -> dict:
            """
            Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
            Args:
                encoded_inputs:
                    Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
                max_length: maximum length of the returned list and optionally padding length (see below).
                    Will truncate by taking into account the special tokens.
                padding_strategy: PaddingStrategy to use for padding.
                    - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                    - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                    - PaddingStrategy.DO_NOT_PAD: Do not pad
                    The tokenizer padding sides are defined in self.padding_side:
                        - 'left': pads on the left of the sequences
                        - 'right': pads on the right of the sequences
                pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                    This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                    >= 7.5 (Volta).
                return_attention_mask:
                    (optional) Set to False to avoid returning attention mask (default: set to model specifics)
            """
            # Load from model defaults
            if return_attention_mask is None:
                return_attention_mask = "attention_mask" in self.model_input_names
            
            # if return_xattn_mask is None:
            #     return_xattn_mask = "cross_attention_mask" in self.model_input_names
            #     print("return_xattn_mask =", return_xattn_mask)
    
            required_input = encoded_inputs[self.model_input_names[0]]
    
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)
    
            if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    
            needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
    
            # Initialize attention mask if not present.
            if return_attention_mask and "attention_mask" not in encoded_inputs:
                encoded_inputs["attention_mask"] = [1] * len(required_input)
                
            # if return_xattn_mask and "cross_attention_mask" not in encoded_inputs:
            #     encoded_inputs["cross_attention_mask"] = [1] * len(required_input)
    
            if needs_to_be_padded:
                difference = max_length - len(required_input)
    
                if self.padding_side == "right":
                    if return_attention_mask:
                        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                    
                    if "cross_attention_mask" in encoded_inputs:
                        encoded_inputs["cross_attention_mask"] = encoded_inputs["cross_attention_mask"] + [0] * difference
                        # print("Xattn after padding (", len(encoded_inputs["cross_attention_mask"]), ")")
                        # print(encoded_inputs["cross_attention_mask"])
                        # print()
                    if "token_type_ids" in encoded_inputs:
                        encoded_inputs["token_type_ids"] = (
                            encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                        )
                    if "special_tokens_mask" in encoded_inputs:
                        encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                    encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
                elif self.padding_side == "left":
                    if return_attention_mask:
                        encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                    if "token_type_ids" in encoded_inputs:
                        encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                            "token_type_ids"
                        ]
                    if "special_tokens_mask" in encoded_inputs:
                        encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                    encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))
            # print("Encoded inputs:", encoded_inputs)
            # print()
            return encoded_inputs


class MT5WithXMask(MT5ForConditionalGeneration):
    """
    This updates the MT5 class to allow the specification of a cross_attention_mask during
    the forward() call.
    """
    
    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_mask: Optional[torch.FloatTensor] = None,
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
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=cross_attention_mask if cross_attention_mask is not None else attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

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
    