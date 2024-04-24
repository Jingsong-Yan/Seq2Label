import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, shift_tokens_right


def double_cross_entropy(inputs, positive_target, negative_target, epsilon=0, reduction="mean"):
    """

    Args:

        inputs:  torch.size([-1, class])
        positive_target:  torch.size([-1])
        negative_target: torch.size([-1])
        epsilon: a
        reduction: "mean" or "sum"

    Returns:
        loss: tensor
    """

    exp = torch.exp(inputs)
    positive_temp = exp.gather(1, positive_target.unsqueeze(-1)).squeeze()
    negative_temp = exp.gather(1, negative_target.unsqueeze(-1)).squeeze()
    sum_temp = exp.sum(1) + epsilon * negative_temp
    epsilon_softmax = positive_temp / sum_temp

    loss = -torch.log(epsilon_softmax)

    if reduction == "mean":
        return loss.mean()
    else:
        return loss.sum()


def cross_entropy(inputs, positive_target, reduction="mean"):
    exp = torch.exp(inputs)
    positive_temp = exp.gather(1, positive_target.unsqueeze(-1)).squeeze()
    sum_temp = exp.sum(1)
    positive_softmax = positive_temp / sum_temp

    positive_log = -torch.log(positive_softmax)

    loss = positive_log
    if reduction == "mean":
        return loss.mean()
    else:
        return loss.sum()


class BartForLabelGeneration(BartPretrainedModel):
    # base_model_prefix = "model"
    # _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, my_config, label_emb=None):

        super().__init__(config)
        self.my_config = my_config
        self.model = BartModel(config)
        if label_emb is None:
            label_emb = nn.Embedding(self.my_config['label_size'], config.d_model,
                                     padding_idx=self.my_config['padding_idx'])
        self.model.get_decoder().set_input_embeddings(label_emb)
        self.lm_head = nn.Linear(config.d_model, self.my_config['label_size'], bias=False)
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    # def get_output_embeddings(self):
    #     return self.lm_head
    #
    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            negative_sample=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if self.my_config['negative_sample']:
            if labels is not None and negative_sample is not None:
                masked_lm_loss = double_cross_entropy(lm_logits.view(-1, self.my_config['label_size']), labels.view(-1),
                                                      negative_sample.view(-1), epsilon=self.my_config['epsilon'])
        else:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.my_config['label_size']), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
