""" Official evaluation script for v1.1 of the SQuAD dataset. [Changed name for external importing]"""
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def span_len(span):
    return span[1] - span[0]

def span_overlap(s1, s2):
    start = max(s1[0], s2[0])
    stop = min(s1[1], s2[1])
    if stop > start:
        return start, stop
    return None

def span_prec(true_span, pred_span):
    overlap = span_overlap(true_span, pred_span)
    if overlap is None:
        return 0.
    return span_len(overlap) / span_len(pred_span)

def span_recall(true_span, pred_span):
    overlap = span_overlap(true_span, pred_span)
    if overlap is None:
        return 0.
    return span_len(overlap) / span_len(true_span)

def span_f1(true_span, pred_span):
    p = span_prec(true_span, pred_span)
    r = span_recall(true_span, pred_span)
    if p == 0 or r == 0:
        return 0.0
    return 2. * p * r / (p + r)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    missing_count = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    missing_count += 1
                    # message = 'Unanswered question ' + qa['id'] + \
                    #           ' will receive score 0.'
                    # print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / (total-missing_count)
    f1 = 100.0 * f1 / (total-missing_count)
    print("missing prediction on %d examples" % (missing_count))
    return {'exact_match': exact_match, 'f1': f1}


def merge_eval(main_eval, new_eval):
  for k in new_eval:
    main_eval['%s' % (k)] = new_eval[k]


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        # if (dataset_json['version'] != expected_version):
        #     print('Evaluation expects v-' + expected_version +
        #           ', but got dataset with v-' + dataset_json['version'],
        #           file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))

    # prediction = '1854–1855'
    # ground_truths = ['1854']
    # print(metric_max_over_ground_truths(
    #     f1_score, prediction, ground_truths))
