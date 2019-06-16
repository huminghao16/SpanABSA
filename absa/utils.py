import json
import collections
import numpy as np

import bert.tokenization as tokenization
from squad.squad_utils import  get_final_text, _get_best_indexes
from squad.squad_evaluate import exact_match_score, f1_score

label_to_id = {'other': 0, 'neutral': 1, 'positive': 2, 'negative': 3, 'conflict': 4}
id_to_label = {0: 'other', 1: 'neutral', 2: 'positive', 3: 'negative', 4: 'conflict'}


class SemEvalExample(object):
    def __init__(self,
                 example_id,
                 sent_tokens,
                 term_texts=None,
                 start_positions=None,
                 end_positions=None,
                 polarities=None):
        self.example_id = example_id
        self.sent_tokens = sent_tokens
        self.term_texts = term_texts
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.polarities = polarities

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        # s += "example_id: %s" % (tokenization.printable_text(self.example_id))
        s += ", sent_tokens: [%s]" % (" ".join(self.sent_tokens))
        if self.term_texts:
            s += ", term_texts: {}".format(self.term_texts)
        # if self.start_positions:
        #     s += ", start_positions: {}".format(self.start_positions)
        # if self.end_positions:
        #     s += ", end_positions: {}".format(self.end_positions)
        if self.polarities:
            s += ", polarities: {}".format(self.polarities)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_positions=None,
                 end_positions=None,
                 start_indexes=None,
                 end_indexes=None,
                 bio_labels=None,
                 polarity_positions=None,
                 polarity_labels=None,
                 label_masks=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.start_indexes = start_indexes
        self.end_indexes = end_indexes
        self.bio_labels = bio_labels
        self.polarity_positions = polarity_positions
        self.polarity_labels = polarity_labels
        self.label_masks = label_masks


def convert_examples_to_features(examples, tokenizer, max_seq_length, verbose_logging=False, logger=None):
    max_term_num = max([len(example.term_texts) for (example_index, example) in enumerate(examples)])
    max_sent_length, max_term_length = 0, 0

    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(examples):
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.sent_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        if len(all_doc_tokens) > max_sent_length:
            max_sent_length = len(all_doc_tokens)

        tok_start_positions = []
        tok_end_positions = []
        for start_position, end_position in \
                zip(example.start_positions, example.end_positions):
            tok_start_position = orig_to_tok_index[start_position]
            if end_position < len(example.sent_tokens) - 1:
                tok_end_position = orig_to_tok_index[end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            tok_start_positions.append(tok_start_position)
            tok_end_positions.append(tok_end_position)

        # Account for [CLS] and [SEP] with "- 2"
        if len(all_doc_tokens) > max_seq_length - 2:
            all_doc_tokens = all_doc_tokens[0:(max_seq_length - 2)]

        tokens = []
        token_to_orig_map = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)

        for index, token in enumerate(all_doc_tokens):
            token_to_orig_map[len(tokens)] = tok_to_orig_index[index]
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # For distant supervision, we annotate the positions of all answer spans
        start_positions = [0] * len(input_ids)
        end_positions = [0] * len(input_ids)
        bio_labels = [0] * len(input_ids)
        polarity_positions = [0] * len(input_ids)
        start_indexes, end_indexes = [], []
        for tok_start_position, tok_end_position, polarity in zip(tok_start_positions, tok_end_positions, example.polarities):
            if (tok_start_position >= 0 and tok_end_position <= (max_seq_length - 1)):
                start_position = tok_start_position + 1   # [CLS]
                end_position = tok_end_position + 1   # [CLS]
                start_positions[start_position] = 1
                end_positions[end_position] = 1
                start_indexes.append(start_position)
                end_indexes.append(end_position)
                term_length = tok_end_position - tok_start_position + 1
                max_term_length = term_length if term_length > max_term_length else max_term_length
                bio_labels[start_position] = 1  # 'B'
                if start_position < end_position:
                    for idx in range(start_position + 1, end_position + 1):
                        bio_labels[idx] = 2  # 'I'
                for idx in range(start_position, end_position + 1):
                    polarity_positions[idx] = label_to_id[polarity]

        polarity_labels = [label_to_id[polarity] for polarity in example.polarities]
        label_masks = [1] * len(polarity_labels)

        while len(start_indexes) < max_term_num:
            start_indexes.append(0)
            end_indexes.append(0)
            polarity_labels.append(0)
            label_masks.append(0)

        assert len(start_indexes) == max_term_num
        assert len(end_indexes) == max_term_num
        assert len(polarity_labels) == max_term_num
        assert len(label_masks) == max_term_num

        if example_index < 1 and verbose_logging:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("tokens: {}".format(tokens))
            logger.info("token_to_orig_map: {}".format(token_to_orig_map))
            logger.info("start_indexes: {}".format(start_indexes))
            logger.info("end_indexes: {}".format(end_indexes))
            logger.info("bio_labels: {}".format(bio_labels))
            logger.info("polarity_positions: {}".format(polarity_positions))
            logger.info("polarity_labels: {}".format(polarity_labels))

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                start_indexes=start_indexes,
                end_indexes=end_indexes,
                bio_labels=bio_labels,
                polarity_positions=polarity_positions,
                polarity_labels=polarity_labels,
                label_masks=label_masks))
        unique_id += 1
    logger.info("Max sentence length: {}".format(max_sent_length))
    logger.info("Max term length: {}".format(max_term_length))
    logger.info("Max term num: {}".format(max_term_num))
    return features


RawSpanResult = collections.namedtuple("RawSpanResult",
                                       ["unique_id", "start_logits", "end_logits"])

RawSpanCollapsedResult = collections.namedtuple("RawSpanCollapsedResult",
                                       ["unique_id", "neu_start_logits", "neu_end_logits", "pos_start_logits", "pos_end_logits",
                                        "neg_start_logits", "neg_end_logits"])

RawBIOResult = collections.namedtuple("RawBIOResult", ["unique_id", "bio_pred"])

RawBIOClsResult = collections.namedtuple("RawBIOClsResult", ["unique_id", "start_indexes", "end_indexes", "bio_pred", "span_masks"])

RawFinalResult = collections.namedtuple("RawFinalResult",
                                        ["unique_id", "start_indexes", "end_indexes", "cls_pred", "span_masks"])


def wrapped_get_final_text(example, feature, start_index, end_index, do_lower_case, verbose_logging, logger):
    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    orig_doc_start = feature.token_to_orig_map[start_index]
    orig_doc_end = feature.token_to_orig_map[end_index]
    orig_tokens = example.sent_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)

    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, logger)
    return final_text


def span_annotate_candidates(all_examples, batch_features, batch_results, filter_type, is_training, use_heuristics, use_nms,
                             logit_threshold, n_best_size, max_answer_length, do_lower_case, verbose_logging, logger):
    """Annotate top-k candidate answers into features."""
    unique_id_to_result = {}
    for result in batch_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    batch_span_starts, batch_span_ends, batch_labels, batch_label_masks = [], [], [], []
    for (feature_index, feature) in enumerate(batch_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        seen_predictions = {}
        span_starts, span_ends, labels, label_masks = [], [], [], []
        if is_training:
            # add ground-truth terms
            for start_index, end_index, polarity_label, mask in \
                    zip(feature.start_indexes, feature.end_indexes, feature.polarity_labels, feature.label_masks):
                if mask and start_index in feature.token_to_orig_map and end_index in feature.token_to_orig_map:
                    final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                        do_lower_case, verbose_logging, logger)
                    if final_text in seen_predictions:
                        continue
                    seen_predictions[final_text] = True

                    span_starts.append(start_index)
                    span_ends.append(end_index)
                    labels.append(polarity_label)
                    label_masks.append(1)
        else:
            prelim_predictions_per_feature = []
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_logit = result.start_logits[start_index]
                    end_logit = result.end_logits[end_index]
                    if start_logit + end_logit < logit_threshold:
                        continue

                    prelim_predictions_per_feature.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=start_logit,
                            end_logit=end_logit))

            if use_heuristics:
                prelim_predictions_per_feature = sorted(
                    prelim_predictions_per_feature,
                    key=lambda x: (x.start_logit + x.end_logit - (x.end_index - x.start_index + 1)),
                    reverse=True)
            else:
                prelim_predictions_per_feature = sorted(
                    prelim_predictions_per_feature,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)

            for i, pred_i in enumerate(prelim_predictions_per_feature):
                if len(span_starts) >= int(n_best_size)/2:
                    break
                final_text = wrapped_get_final_text(example, feature, pred_i.start_index, pred_i.end_index,
                                                    do_lower_case, verbose_logging, logger)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True

                span_starts.append(pred_i.start_index)
                span_ends.append(pred_i.end_index)
                labels.append(0)
                label_masks.append(1)

                # filter out redundant candidates
                if (i+1) < len(prelim_predictions_per_feature) and use_nms:
                    indexes = []
                    for j, pred_j in enumerate(prelim_predictions_per_feature[(i+1):]):
                        filter_text = wrapped_get_final_text(example, feature, pred_j.start_index, pred_j.end_index,
                                                             do_lower_case, verbose_logging, logger)
                        if filter_type == 'em':
                            if exact_match_score(final_text, filter_text):
                                indexes.append(i + j + 1)
                        elif filter_type == 'f1':
                            if f1_score(final_text, filter_text) > 0:
                                indexes.append(i + j + 1)
                        else:
                            raise Exception
                    [prelim_predictions_per_feature.pop(index - k) for k, index in enumerate(indexes)]

        # Pad to fixed length
        while len(span_starts) < int(n_best_size):
            span_starts.append(0)
            span_ends.append(0)
            labels.append(0)
            label_masks.append(0)
        assert len(span_starts) == int(n_best_size)
        assert len(span_ends) == int(n_best_size)
        assert len(labels) == int(n_best_size)
        assert len(label_masks) == int(n_best_size)

        batch_span_starts.append(span_starts)
        batch_span_ends.append(span_ends)
        batch_labels.append(labels)
        batch_label_masks.append(label_masks)
    return batch_span_starts, batch_span_ends, batch_labels, batch_label_masks


def ts2start_end(ts_tag_sequence):
    starts, ends = [], []
    n_tag = len(ts_tag_sequence)
    prev_pos, prev_sentiment = '$$$', '$$$'
    tag_on = False
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag != 'O':
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
        else:
            cur_pos, cur_sentiment = 'O', '$$$'
        assert cur_pos == 'O' or cur_pos == 'T'
        if cur_pos == 'T':
            if prev_pos != 'T':
                # cur tag is at the beginning of the opinion target
                starts.append(i)
                tag_on = True
            else:
                if cur_sentiment != prev_sentiment:
                    # prev sentiment is not equal to current sentiment
                    ends.append(i - 1)
                    starts.append(i)
                    tag_on = True
        else:
            if prev_pos == 'T':
                ends.append(i - 1)
                tag_on = False
        prev_pos = cur_pos
        prev_sentiment = cur_sentiment
    if tag_on:
        ends.append(n_tag-1)
    assert len(starts) == len(ends), (len(starts), len(ends), ts_tag_sequence)
    return starts, ends


def ts2polarity(words, ts_tag_sequence, starts, ends):
    polarities = []
    for start, end in zip(starts, ends):
        cur_ts_tag = ts_tag_sequence[start]
        cur_pos, cur_sentiment = cur_ts_tag.split('-')
        assert cur_pos == 'T'
        prev_sentiment = cur_sentiment
        if start < end:
            for idx in range(start, end + 1):
                cur_ts_tag = ts_tag_sequence[idx]
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                assert cur_pos == 'T'
                assert cur_sentiment == prev_sentiment, (words, ts_tag_sequence, start, end)
                prev_sentiment = cur_sentiment
        polarities.append(cur_sentiment)
    return polarities


def pos2term(words, starts, ends):
    term_texts = []
    for start, end in zip(starts, ends):
        term_texts.append(' '.join(words[start:end+1]))
    return term_texts


def convert_absa_data(dataset, verbose_logging=False):
    examples = []
    n_records = len(dataset)
    for i in range(n_records):
        words = dataset[i]['words']
        ts_tags = dataset[i]['ts_raw_tags']
        starts, ends = ts2start_end(ts_tags)
        polarities = ts2polarity(words, ts_tags, starts, ends)
        term_texts = pos2term(words, starts, ends)

        if term_texts != []:
            new_polarities = []
            for polarity in polarities:
                if polarity == 'POS':
                    new_polarities.append('positive')
                elif polarity == 'NEG':
                    new_polarities.append('negative')
                elif polarity == 'NEU':
                    new_polarities.append('neutral')
                else:
                    raise Exception
            assert len(term_texts) == len(starts)
            assert len(term_texts) == len(new_polarities)
            example = SemEvalExample(str(i), words, term_texts, starts, ends, new_polarities)
            examples.append(example)
            if i < 50 and verbose_logging:
                print(example)
    print("Convert %s examples" % len(examples))
    return examples


def read_absa_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # tag sequence for opinion target extraction
            ote_tags = []
            # word sequence
            words = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                words.append(word.lower())
                if tag == 'O':
                    ote_tags.append('O')
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ote_tags.append('T')
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ote_tags.append('T')
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ote_tags.append('T')
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['ote_raw_tags'] = ote_tags.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset