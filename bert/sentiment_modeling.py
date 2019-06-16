from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from bert.modeling import BertModel, BERTLayerNorm
from allennlp.modules import ConditionalRandomField

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

def pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(0)
    return sequence

def convert_crf_output(outputs, sequence_length, device):
    predictions = []
    for output in outputs:
        pred = pad_sequence(output[0], sequence_length)
        predictions.append(torch.tensor(pred, dtype=torch.long))
    predictions = torch.stack(predictions, dim=0)
    if device is not None:
        predictions = predictions.to(device)
    return predictions


class BertForBIOAspectExtraction(nn.Module):
    def __init__(self, config, use_crf=False):
        super(BertForBIOAspectExtraction, self).__init__()
        self.bert = BertModel(config)
        self.use_crf = use_crf
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.affine = nn.Linear(config.hidden_size, 3)
        if self.use_crf:
            self.crf = ConditionalRandomField(3)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, bio_labels=None, device=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.affine(sequence_output)   # [N, L, 3]

        if bio_labels is not None:
            if self.use_crf:
                total_loss = -self.crf(logits, bio_labels, attention_mask) / (logits.size()[0])
            else:
                flat_logits = flatten(logits)
                bio_labels = flatten(bio_labels)
                attention_mask = flatten(attention_mask).to(dtype=flat_logits.dtype)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(flat_logits, bio_labels)
                total_loss = torch.sum(attention_mask * loss) / attention_mask.sum()
            return total_loss
        else:
            if self.use_crf:
                outputs = self.crf.viterbi_tags(logits, attention_mask)
                sequence_length = logits.size()[1]
                predictions = convert_crf_output(outputs, sequence_length, device)
                return predictions
            else:
                return logits


class BertForBIOAspectClassification(nn.Module):
    def __init__(self, config, use_crf=False):
        super(BertForBIOAspectClassification, self).__init__()
        self.bert = BertModel(config)
        self.use_crf = use_crf
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.affine = nn.Linear(config.hidden_size, 5)
        if self.use_crf:
            self.crf = ConditionalRandomField(5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, polarity_positions=None, device=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.affine(sequence_output)   # [N, L, 5]

        if polarity_positions is not None:
            if self.use_crf:
                total_loss = -self.crf(logits, polarity_positions, attention_mask) / (logits.size()[0])
            else:
                flat_logits = flatten(logits)
                flat_polarity_positions = flatten(polarity_positions)
                attention_mask = flatten(attention_mask).to(dtype=flat_logits.dtype)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(flat_logits, flat_polarity_positions)
                total_loss = torch.sum(attention_mask * loss) / attention_mask.sum()
            return total_loss
        else:
            if self.use_crf:
                outputs = self.crf.viterbi_tags(logits, attention_mask)
                sequence_length = logits.size()[1]
                predictions = convert_crf_output(outputs, sequence_length, device)
                return predictions
            else:
                return logits


class BertForSpanAspectExtraction(nn.Module):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForSpanAspectExtraction, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class BertForSpanAspectClassification(nn.Module):
    def __init__(self, config):
        super(BertForSpanAspectClassification, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.affine = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, 5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, span_starts=None, span_ends=None,
                labels=None, label_masks=None):
        '''
        :param input_ids: [N, L]
        :param token_type_ids: [N, L]
        :param attention_mask: [N, L]
        :param span_starts: [N, M]
        :param span_ends: [N, M]
        :param labels: [N, M]
        '''
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert span_starts is not None and span_ends is not None and labels is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            cls_logits = self.classifier(span_pooled_output)  # [N*M, 4]

            cls_loss_fct = CrossEntropyLoss(reduction='none')
            flat_cls_labels = flatten(labels)
            flat_label_masks = flatten(label_masks)
            loss = cls_loss_fct(cls_logits, flat_cls_labels)
            mean_loss = torch.sum(loss * flat_label_masks.to(dtype=loss.dtype)) / torch.sum(flat_label_masks.to(dtype=loss.dtype))
            return mean_loss

        elif mode == 'inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert span_starts is not None and span_ends is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            cls_logits = self.classifier(span_pooled_output)  # [N*M, 4]
            return reconstruct(cls_logits, span_starts)

        else:
            raise Exception


class BertForJointBIOExtractAndClassification(nn.Module):
    def __init__(self, config, use_crf=False):
        super(BertForJointBIOExtractAndClassification, self).__init__()
        self.bert = BertModel(config)
        self.use_crf = use_crf
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bio_affine = nn.Linear(config.hidden_size, 3)
        self.cls_affine = nn.Linear(config.hidden_size, 5)
        if self.use_crf:
            self.cls_crf = ConditionalRandomField(5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None,
                bio_labels=None, polarity_positions=None, sequence_input=None, device=None):
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert bio_labels is not None
            ae_logits = self.bio_affine(sequence_output)
            flat_ae_logits = flatten(ae_logits)
            flat_bio_labels = flatten(bio_labels)
            flat_attention_mask = flatten(attention_mask).to(dtype=flat_ae_logits.dtype)
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(flat_ae_logits, flat_bio_labels)
            ae_loss = torch.sum(flat_attention_mask * loss) / flat_attention_mask.sum()

            assert polarity_positions is not None
            ac_logits = self.cls_affine(sequence_output)
            if self.use_crf:
                ac_loss = -self.cls_crf(ac_logits, polarity_positions, attention_mask) / (ac_logits.size()[0])
            else:
                flat_ac_logits = flatten(ac_logits)
                flat_polarity_positions = flatten(polarity_positions)
                flat_attention_mask = flatten(attention_mask).to(dtype=flat_ac_logits.dtype)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(flat_ac_logits, flat_polarity_positions)
                ac_loss = torch.sum(flat_attention_mask * loss) / flat_attention_mask.sum()

            return ae_loss + ac_loss

        elif mode == 'extract_inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]
            ae_logits = self.bio_affine(sequence_output)
            return ae_logits, sequence_output

        elif mode == 'classify_inference':
            assert sequence_input is not None
            ac_logits = self.cls_affine(sequence_input)

            if self.use_crf:
                outputs = self.cls_crf.viterbi_tags(ac_logits, attention_mask)
                sequence_length = ac_logits.size()[1]
                predictions = convert_crf_output(outputs, sequence_length, device)
                return predictions
            else:
                return ac_logits


class BertForJointSpanExtractAndClassification(nn.Module):
    def __init__(self, config):
        super(BertForJointSpanExtractAndClassification, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.unary_affine = nn.Linear(config.hidden_size, 1)
        self.binary_affine = nn.Linear(config.hidden_size, 2)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, start_positions=None, end_positions=None,
                span_starts=None, span_ends=None, polarity_labels=None, label_masks=None, sequence_input=None):
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert start_positions is not None and end_positions is not None
            ae_logits = self.binary_affine(sequence_output)   # [N, L, 2]
            start_logits, end_logits = ae_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            ae_loss = (start_loss + end_loss) / 2

            assert span_starts is not None and span_ends is not None and polarity_labels is not None \
                   and label_masks is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            ac_loss_fct = CrossEntropyLoss(reduction='none')
            flat_polarity_labels = flatten(polarity_labels)
            flat_label_masks = flatten(label_masks).to(dtype=ac_logits.dtype)
            ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
            ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()

            return ae_loss + ac_loss

        elif mode == 'extract_inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]
            ae_logits = self.binary_affine(sequence_output)
            start_logits, end_logits = ae_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return start_logits, end_logits, sequence_output

        elif mode == 'classify_inference':
            assert span_starts is not None and span_ends is not None and sequence_input is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]

            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            return reconstruct(ac_logits, span_starts)


class BertForCollapsedBIOAspectExtractionAndClassification(nn.Module):
    def __init__(self, config, use_crf=False):
        super(BertForCollapsedBIOAspectExtractionAndClassification, self).__init__()
        self.bert = BertModel(config)
        self.use_crf = use_crf
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.affine = nn.Linear(config.hidden_size, 7)
        if self.use_crf:
            self.crf = ConditionalRandomField(7)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, bio_labels=None, device=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.affine(sequence_output)   # [N, L, 7]

        if bio_labels is not None:  # [N, L]
            if self.use_crf:
                total_loss = -self.crf(logits, bio_labels, attention_mask) / (logits.size()[0])
            else:
                flat_logits = flatten(logits)   # [N*L, 3]
                bio_labels = flatten(bio_labels)    # [N*L]
                attention_mask = flatten(attention_mask).to(dtype=flat_logits.dtype)    # [N*L]
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(flat_logits, bio_labels) # [N*L]
                total_loss = torch.sum(attention_mask * loss) / attention_mask.sum()
            return total_loss
        else:
            if self.use_crf:
                outputs = self.crf.viterbi_tags(logits, attention_mask)
                sequence_length = logits.size()[1]
                predictions = convert_crf_output(outputs, sequence_length, device)
                return predictions
            else:
                return logits


class BertForCollapsedSpanAspectExtractionAndClassification(nn.Module):
    def __init__(self, config):
        super(BertForCollapsedSpanAspectExtractionAndClassification, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.neu_outputs = nn.Linear(config.hidden_size, 2)
        self.pos_outputs = nn.Linear(config.hidden_size, 2)
        self.neg_outputs = nn.Linear(config.hidden_size, 2)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, neu_start_positions=None, neu_end_positions=None,
                pos_start_positions=None, pos_end_positions=None, neg_start_positions=None, neg_end_positions=None,
                neu_mask=None, pos_mask=None, neg_mask=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        neu_logits = self.neu_outputs(sequence_output)
        neu_start_logits, neu_end_logits = neu_logits.split(1, dim=-1)
        neu_start_logits = neu_start_logits.squeeze(-1)
        neu_end_logits = neu_end_logits.squeeze(-1)
        
        pos_logits = self.pos_outputs(sequence_output)
        pos_start_logits, pos_end_logits = pos_logits.split(1, dim=-1)
        pos_start_logits = pos_start_logits.squeeze(-1)
        pos_end_logits = pos_end_logits.squeeze(-1)

        neg_logits = self.neg_outputs(sequence_output)
        neg_start_logits, neg_end_logits = neg_logits.split(1, dim=-1)
        neg_start_logits = neg_start_logits.squeeze(-1)
        neg_end_logits = neg_end_logits.squeeze(-1)

        if neu_start_positions is not None and neu_end_positions is not None and \
                pos_start_positions is not None and pos_end_positions is not None and \
                neg_start_positions is not None and neg_end_positions is not None and \
                neu_mask is not None and pos_mask is not None and neg_mask is not None:

            neu_loss = distant_loss(neu_start_logits, neu_end_logits, neu_start_positions, neu_end_positions, neu_mask)
            pos_loss = distant_loss(pos_start_logits, pos_end_logits, pos_start_positions, pos_end_positions, pos_mask)
            neg_loss = distant_loss(neg_start_logits, neg_end_logits, neg_start_positions, neg_end_positions, neg_mask)
        
            return neu_loss + pos_loss + neg_loss
        else:
            return neu_start_logits, neu_end_logits, pos_start_logits, pos_end_logits, neg_start_logits, neg_end_logits


def distant_loss(start_logits, end_logits, start_positions=None, end_positions=None, mask=None):
    start_loss = distant_cross_entropy(start_logits, start_positions, mask)
    end_loss = distant_cross_entropy(end_logits, end_positions, mask)
    total_loss = (start_loss + end_loss) / 2
    return total_loss

