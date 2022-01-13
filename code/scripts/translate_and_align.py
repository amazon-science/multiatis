import nlu_constants
import boto3
import csv
import json
import jieba
import sys
import subprocess

lang = sys.argv[1]

data_dir = "../data/"
train_tsv = "atis_train.tsv"
valid_tsv = "atis_dev.tsv"
train_target = "train_translated_" + lang.upper() + ".tsv"
valid_target = "dev_translated_" + lang.upper() + ".tsv"
source_lang = "en"
target_lang = lang


def idx2label(sources, source_labels):
    # we remove the BI tags because the order of BI may be changed after translation.
    ret_token_sls = []  # return a list of dictionary where
    for ix, ex in enumerate(sources):
        lbs = source_labels[ix]
        assert len(ex) == len(lbs)
        sls = {}
        for jx, token in enumerate(ex):
            if lbs[jx] != 'O':
                sls[jx] = lbs[jx][2:]
        ret_token_sls.append(sls)
    return ret_token_sls


def load_tsv(fn):
    sources = []
    source_labels = []
    with open(fn) as tsvFile:
        tsvReader = csv.DictReader(tsvFile, delimiter="\t")
        for ix, line in enumerate(tsvReader):
            sources.append(line[nlu_constants.UTTERANCE_SYMBOL].split(' '))
            source_labels.append(line[nlu_constants.SLOT_LABEL_SYMBOL].split(' '))
    return sources, source_labels


for source_tsv, target_tsv in [(train_tsv, train_target), (valid_tsv, valid_target)]:
    sources, source_labels = load_tsv(data_dir + source_tsv)
    token_sls = idx2label(sources, source_labels)
    with open(data_dir + target_tsv[:-4] + "_idx2label" + ".json", "w") as fw:
        json.dump(token_sls, fw)
    translator = boto3.client(service_name='translate', use_ssl=True, region_name='us-east-1')
    targets = []
    for source in sources:
        result = translator.translate_text(Text=" ".join(source), SourceLanguageCode=source_lang,
                                           TargetLanguageCode=target_lang)
        targets.append(result.get('TranslatedText'))
    with open(data_dir + target_tsv[:-4] + ".json", "w") as fw:
        json.dump(targets, fw)
    with open(data_dir + target_tsv[:-4] + ".json") as f:
        targets = json.load(f)
    tokenized_targets = []
    for target in targets:
        if target_lang == 'zh':
            target = target.replace(",", "")
            segs = [t.strip() for t in list(jieba.cut(target)) if not t.isspace()]
        else:
            target = target.replace(",", " ")
            segs = target.strip().split(' ')
        tokenized_targets.append(segs)
    with open(data_dir + target_tsv[:-4] + "_token" + ".json", "w") as fw:
        json.dump(tokenized_targets, fw)

with open(data_dir + "atis_train_valid", "w") as fw:
    for source_tsv, target_tsv in [(train_tsv, train_target), (valid_tsv, valid_target)]:
        source_utterances, _ = load_tsv(data_dir + source_tsv)
        with open(data_dir + target_tsv[:-4] + "_token" + ".json") as f:
            target_utterances = json.load(f)
        for ix in range(len(source_utterances)):
            fw.write((" ".join(source_utterances[ix]) + " ||| " + " ".join(target_utterances[ix]) + "\n"))

with open(data_dir + "forward.align", "w") as f:
    command = ["./fast_align/build/fast_align", "-i", data_dir + "atis_train_valid", "-d", "-o", "-v"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout = process.communicate()[0].decode('utf-8')
    f.write(stdout)

lens = [0]
for target_tsv in [train_target, valid_target]:
    with open(data_dir + target_tsv[:-4] + "_token" + ".json") as f:
        lens.append(len(json.load(f)) + lens[-1])
lens = lens[1:]

s2t_set = []
with open(data_dir + "forward.align") as f:
    s2t_indexes = []
    len_idx = 0
    for ix, l in enumerate(f):
        if ix == lens[len_idx]:
            s2t_set.append(s2t_indexes)
            s2t_indexes = []
            len_idx += 1
        if ix >= lens[-1]:
            break
        segs = l.split()
        s2t_idx = {}
        for seg in segs:
            st = seg.split('-')
            s2t_idx[int(st[0])] = int(st[1])
        s2t_indexes.append(s2t_idx)
    if len(s2t_indexes) > 0:
        s2t_set.append(s2t_indexes)


def align_labels(source_utterances, target_utterances, s2t_indexes, source_idx2labels):
    # generate target labels.
    ret_target_labels = []
    for ix, tokens in enumerate(source_utterances):
        template = ['O'] * len(target_utterances[ix])  # generate template labels
        for jx in range(len(source_utterances[ix])):
            if jx in s2t_indexes[ix] and str(jx) in source_idx2labels[ix]:
                template[s2t_indexes[ix][jx]] = source_idx2labels[ix][str(jx)]
        # add BI labels
        state = 'O'
        for jx in range(len(template)):
            if template[jx] != 'O' and (state == 'O' or state != template[jx]):
                state = template[jx]
                template[jx] = 'B-' + template[jx]
            elif template[jx] != 'O' and state == template[jx]:
                template[jx] = 'I-' + template[jx]
            elif template[jx] == 'O':
                state = 'O'
        ret_target_labels.append(template)
    return ret_target_labels


for ix, (source_tsv, target_tsv) in enumerate([(train_tsv, train_target), (valid_tsv, valid_target)]):
    source_utterances, source_labels = load_tsv(data_dir + source_tsv)
    with open(data_dir + target_tsv[:-4] + "_token" + ".json") as f:
        target_tokens = json.load(f)
    s2t_indexes = s2t_set[ix]
    with open(data_dir + target_tsv[:-4] + "_idx2label" + ".json") as f:
        token_sls = json.load(f)
    target_labels = align_labels(source_utterances, target_tokens, s2t_indexes, token_sls)
    with open(data_dir + source_tsv) as tsvFile:  # also gen tsv format for DiSAN
        tsvReader = csv.DictReader(tsvFile, delimiter="\t")
        with open(data_dir + target_tsv, "w") as tsvFileW:
            tsvWriter = csv.DictWriter(tsvFileW, fieldnames=tsvReader.fieldnames, delimiter="\t")
            tsvWriter.writeheader()
            for jx, line in enumerate(tsvReader):
                line[nlu_constants.UTTERANCE_SYMBOL] = ' '.join(target_tokens[jx])
                line[nlu_constants.SLOT_LABEL_SYMBOL] = ' '.join(target_labels[jx])
                tsvWriter.writerow(line)
