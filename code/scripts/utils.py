import csv
import mxnet as mx
import numpy as np
import random

from mxnet import gluon

PAD = '[PAD]'

def process_seq_labels(label, pred, ignore_id=-1):
    # label: (batch_size * seq_length)
    label = label.reshape(-3).asnumpy()
    # pred: (batch_size * seq_length, num_labels)
    pred = pred.reshape(-3, 0).asnumpy()
    # ignore ignore_id
    keep_idx = np.where(label != ignore_id)
    label = mx.nd.array(label[keep_idx])
    pred = mx.nd.array(pred[keep_idx])
    return [label], [pred]

def split_and_load(arrs, ctx):
    """split and load arrays to a list of contexts"""
    assert isinstance(arrs, (list, tuple))
    if len(ctx) == 1:
        return [[arr.as_in_context(ctx[0]) for arr in arrs]]
    else:
        # split and load
        loaded_arrs = [gluon.utils.split_and_load(arr, ctx, even_split=False) for arr in arrs]
        return zip(*loaded_arrs)

def label2index(map, key):
    return map[key] if key in map else len(map)

def load_tsv(fn):
    example_ids = []
    utterances = []
    labels = []
    intents = []
    with open(fn) as tsvFile:
        tsvReader = csv.DictReader(tsvFile, delimiter="\t")
        for line in tsvReader:
            example_ids.append(int(line['u_id']))
            utterances.append(line['utterance'].split(' '))
            labels.append(line['slot-labels'].split(' '))
            intents.append(line['intent'])
    return example_ids, utterances, labels, intents

def get_label_indices(input_file):
    _, _, train_labels, train_intents = load_tsv(input_file)

    intent2idx = {}
    for intent in train_intents:
        if intent not in intent2idx:
            intent2idx[intent] = len(intent2idx)

    label2idx = {}
    for labels in train_labels:
        for l in labels:
            if l not in label2idx:
                label2idx[l] = len(label2idx)

    new_labels = []
    for label in label2idx.keys():
        if label.startswith('B'):
            cont_label = 'I' + label[1:]
            if cont_label not in label2idx:
                new_labels.append(cont_label)
    for label in new_labels:
        label2idx[label] = len(label2idx)
    if PAD not in label2idx:
        label2idx[PAD] = len(label2idx)
    return intent2idx, label2idx

def merge_slots(slots, alignment):
    merged_slots = []
    start_idx = alignment[0]
    for end_idx in alignment[1:]:
        tag = slots[start_idx]
        for slot in slots[start_idx: end_idx]:
            if slot.startswith('B') and tag == 'O':
                tag = slot
            elif slot.startswith('I') and tag == 'O':
                tag = slot
        start_idx = end_idx
        merged_slots.append(tag)
    return merged_slots

def icsl_transform(sample, vocabulary, label2idx, intent2idx, bert_tokenizer):
    eid = int(sample[3])
    out_sample = []
    tag_alignment = []
    bert_tokens = ['[CLS]']
    bert_tags = []
    for w, tag in zip(sample[0].split(), sample[1].split()):
        tag_alignment.append(len(bert_tags))
        bert_toks = bert_tokenizer(w)
        bert_tokens.extend(bert_toks)
        if tag.startswith('B'):
            cont_tag = 'I' + tag[1:]
            bert_tags.extend([tag] + [cont_tag] * (len(bert_toks) - 1))
        else:
            bert_tags.extend([tag] * len(bert_toks))
    bert_tokens += ['[SEP]']
    bert_tags += [PAD]
    # add example id
    out_sample += [eid]
    # add token ids
    out_sample += [[vocabulary[tok] for tok in bert_tokens]]
    # add slot labels
    out_sample += [[label2index(label2idx, tag) for tag in bert_tags]]
    # add intent label
    out_sample += [label2index(intent2idx, sample[2])]
    # add valid length
    valid_len = len(bert_tokens)
    out_sample += [valid_len]
    return out_sample, tag_alignment

def parallel_icsl_transform(sample, vocabulary, label2idx, intent2idx, bert_tokenizer):
    out_sample = []
    target = ['[CLS]']
    bert_tags = []
    for w, tag in zip(sample[1].split(), sample[2].split()):
        bert_toks = bert_tokenizer(w)
        target.extend(bert_toks)
        if tag.startswith('B'):
            cont_tag = 'I' + tag[1:]
            bert_tags.extend([tag] + [cont_tag] * (len(bert_toks) - 1))
        else:
            bert_tags.extend([tag] * len(bert_toks))
    target += ['[SEP]']
    bert_tags += [PAD]
    source = ['[CLS]'] + bert_tokenizer(sample[0]) + ['[SEP]']
    # add source ids
    out_sample += [[vocabulary[tok] for tok in source]]
    # add target ids
    out_sample += [[vocabulary[tok] for tok in target]]
    # add slot labels
    out_sample += [[label2index(label2idx, tag) for tag in bert_tags]]
    # add intent label
    out_sample += [label2index(intent2idx, sample[3])]
    # add source valid length
    out_sample += [len(source)]
    # add target valid length
    out_sample += [len(target)]
    return out_sample
