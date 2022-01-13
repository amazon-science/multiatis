import collections
import gluonnlp as nlp
import logging
import mxnet as mx
import os
import sklearn.metrics
import subprocess
import sys
import time
import warnings

from bert import *
from mxnet import gluon
from mxnet.gluon import Block, nn, rnn

from loss import *
from utils import *

random_seed = int(sys.argv[1])
warnings.filterwarnings('ignore')
data_dir = "../data/"
model_dir = "../exp/"
conll_prediction_file = data_dir + "conll.pred"

PAD = '[PAD]'
mx.random.seed(random_seed)
ctx = [mx.gpu(0), mx.gpu(1)]

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='[%(levelname)s] %(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')
fh = logging.FileHandler(os.path.join(model_dir, 'bert.' + str(random_seed) + '.log'), mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

class BERTForICSL(Block):
    """Model for IC/SL task.

    The model feeds token ids into BERT to get the sequence
    representations, then apply two dense layers for IC/SL task.
    """

    def __init__(self, bert, num_slot_labels, num_intents, hidden_size=768, dropout=.1, prefix=None, params=None):
        super(BERTForICSL, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.dropout = nn.Dropout(rate=dropout)
            # IC/SL classifier
            self.slot_classifier = nn.Dense(units=num_slot_labels,
                                            in_units=hidden_size,
                                            flatten=False)
            self.intent_classifier = nn.Dense(units=num_intents,
                                              in_units=hidden_size,
                                              flatten=False)

    def encode(self, inputs, valid_length):
        types = mx.nd.zeros_like(inputs)
        encoded = self.bert(inputs, types, valid_length)
        encoded = self.dropout(encoded)
        return encoded

    def forward(self, inputs, valid_length):  # pylint: disable=arguments-differ
        """Generate unnormalized scores for the given input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        valid_length : NDArray or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        intent_prediction: NDArray
            Shape (batch_size, num_intents)
        slot_prediction : NDArray
            Shape (batch_size, seq_length, num_slot_labels)
        """
        # hidden: (batch_size, seq_length, hidden_size)
        hidden = self.encode(inputs, valid_length)
        # get intent and slot label predictions
        intent_prediction = self.intent_classifier(hidden[:, 0, :])
        slot_prediction = self.slot_classifier(hidden[:, 1:, :])
        return intent_prediction, slot_prediction


def train(model_name, train_input, dev_input):
    """Training function."""
    ## Arguments
    log_interval = 100
    batch_size = 32
    lr = 1e-5
    optimizer = 'adam'
    accumulate = None
    epochs = 20

    ## Load BERT model and vocabulary
    bert, vocabulary = nlp.model.get_model('bert_12_768_12',
                                           dataset_name='wiki_multilingual_uncased',
                                           pretrained=True,
                                           ctx=ctx,
                                           use_pooler=False,
                                           use_decoder=False,
                                           use_classifier=False)

    model = BERTForICSL(bert, num_slot_labels=len(label2idx), num_intents=len(intent2idx))
    model.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
    model.hybridize(static_alloc=True)

    icsl_loss_function = ICSLLoss()
    icsl_loss_function.hybridize(static_alloc=True)

    ic_metric = mx.metric.Accuracy()
    sl_metric = mx.metric.Accuracy()

    ## Load labeled data
    field_separator = nlp.data.Splitter('\t')
    # fields to select from the file: utterance, slot labels, intent, uid
    field_indices = [1, 3, 4, 0]
    train_data = nlp.data.TSVDataset(filename=train_input,
                                     field_separator=field_separator,
                                     num_discard_samples=1,
                                     field_indices=field_indices)

    # use the vocabulary from pre-trained model for tokenization
    bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
    train_data_transform = train_data.transform(fn=lambda x: icsl_transform(x, vocabulary, label2idx, intent2idx, bert_tokenizer)[0])
    # create data loader
    pad_token_id = vocabulary[PAD]
    pad_label_id = label2idx[PAD]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=pad_token_id),
        nlp.data.batchify.Pad(axis=0, pad_val=pad_label_id),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))
    train_sampler = nlp.data.FixedBucketSampler(lengths=[len(item[1]) for item in train_data_transform],
                                                batch_size=batch_size,
                                                shuffle=True)
    train_dataloader = mx.gluon.data.DataLoader(train_data_transform,
                                                batchify_fn=batchify_fn,
                                                batch_sampler=train_sampler)

    optimizer_params = {'learning_rate': lr}
    trainer = gluon.Trainer(model.collect_params(), optimizer,
                            optimizer_params, update_on_kvstore=False)

    # Collect differentiable parameters
    params = [p for p in model.collect_params().values() if p.grad_req != 'null']
    # Set grad_req if gradient accumulation is required
    if accumulate:
        for p in params:
            p[1].grad_req = 'add'
    # Fix BERT embeddings if required
    for p in model.collect_params().items():
        if 'embed' in p[0]:
            p[1].grad_req = 'null'

    epoch_tic = time.time()
    total_num = 0
    log_num = 0
    best_score = (0, 0)
    for epoch_id in range(epochs):
        step_loss = 0
        tic = time.time()
        # train on labeled data
        for batch_id, data in enumerate(train_dataloader):
            # forward and backward
            with mx.autograd.record():
                if data[0].shape[0] < len(ctx):
                    data = split_and_load(data, [ctx[0]])
                else:
                    data = split_and_load(data, ctx)
                for chunk in data:
                    _, token_ids, slot_label, intent_label, valid_length = chunk

                    log_num += len(token_ids)
                    total_num += len(token_ids)

                    # forward computation
                    intent_pred, slot_pred = model(token_ids, valid_length)
                    ls = icsl_loss_function(intent_pred, slot_pred, intent_label, slot_label, valid_length - 2).mean()

                    if accumulate:
                        ls = ls / accumulate
                    ls.backward()
                    step_loss += ls.asscalar()

            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1, ignore_stale_grad=True)

            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                # update metrics
                ic_metric.update([intent_label], [intent_pred])
                sl_metric.update(*process_seq_labels(slot_label, slot_pred, ignore_id=pad_label_id))
                log.info('Epoch: {}, Batch: {}/{}, speed: {:.2f} samples/s, lr={:.7f}, loss={:.4f}, intent acc={:.3f}, slot acc={:.3f}'
                         .format(epoch_id,
                                 batch_id,
                                 len(train_dataloader),
                                 log_num / (toc - tic),
                                 trainer.learning_rate,
                                 step_loss / log_interval,
                                 ic_metric.get()[1],
                                 sl_metric.get()[1]))
                tic = time.time()
                step_loss = 0
                log_num = 0

        mx.nd.waitall()
        epoch_toc = time.time()
        log.info('Time cost: {:.2f} s, Speed: {:.2f} samples/s'
                 .format(epoch_toc - epoch_tic, total_num/(epoch_toc - epoch_tic)))
        # evaluate on development set
        log.info('Evaluate on development set:')
        intent_acc, slot_f1 = evaluate(model=model, eval_input=dev_input)
        if slot_f1 > best_score[1]:
            best_score = (intent_acc, slot_f1)
            model.save_parameters(os.path.join(model_dir, model_name + '.params'))


def evaluate(model=None, model_name='', eval_input=''):
    """Evaluate the model on validation dataset.
    """
    ## Load model
    bert, vocabulary = nlp.model.get_model('bert_12_768_12',
                                           dataset_name='wiki_multilingual_uncased',
                                           pretrained=True,
                                           ctx=ctx,
                                           use_pooler=False,
                                           use_decoder=False,
                                           use_classifier=False)
    if model is None:
        assert model_name != ''
        model = BERTForICSL(bert, num_slot_labels=len(label2idx), num_intents=len(intent2idx))
        model.initialize(ctx=ctx)
        model.hybridize(static_alloc=True)
        model.load_parameters(os.path.join(model_dir, model_name + '.params'))

    idx2label = {}
    for label, idx in label2idx.items():
        idx2label[idx] = label
    ## Load dev dataset
    field_separator = nlp.data.Splitter('\t')
    field_indices = [1, 3, 4, 0]
    eval_data = nlp.data.TSVDataset(filename=eval_input,
                                    field_separator=field_separator,
                                    num_discard_samples=1,
                                    field_indices=field_indices)

    bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

    dev_alignment = {}
    eval_data_transform = []
    for sample in eval_data:
        sample, alignment = icsl_transform(sample, vocabulary, label2idx, intent2idx, bert_tokenizer)
        eval_data_transform += [sample]
        dev_alignment[sample[0]] = alignment
    log.info('The number of examples after preprocessing: {}'
             .format(len(eval_data_transform)))

    test_batch_size = 16
    pad_token_id = vocabulary[PAD]
    pad_label_id = label2idx[PAD]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=pad_token_id),
        nlp.data.batchify.Pad(axis=0, pad_val=pad_label_id),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))
    eval_dataloader = mx.gluon.data.DataLoader(
        eval_data_transform,
        batchify_fn=batchify_fn,
        num_workers=4, batch_size=test_batch_size, shuffle=False, last_batch='keep')

    _Result = collections.namedtuple(
        '_Result', ['intent', 'slot_labels'])
    all_results = {}

    total_num = 0
    for data in eval_dataloader:
        example_ids, token_ids, _, _, valid_length = data
        total_num += len(token_ids)
        # load data to GPU
        token_ids = token_ids.astype('float32').as_in_context(ctx[0])
        valid_length = valid_length.astype('float32').as_in_context(ctx[0])

        # forward computation
        intent_pred, slot_pred = model(token_ids, valid_length)
        intent_pred = intent_pred.asnumpy()
        slot_pred = slot_pred.asnumpy()
        valid_length = valid_length.asnumpy()

        for eid, y_intent, y_slot, length in zip(example_ids, intent_pred, slot_pred, valid_length):
            eid = eid.asscalar()
            length = int(length) - 2
            intent_id = y_intent.argmax(axis=-1)
            slot_ids = y_slot.argmax(axis=-1).tolist()[:length]
            slot_names = [idx2label[idx] for idx in slot_ids]
            merged_slot_names = merge_slots(slot_names, dev_alignment[eid] + [length])
            if eid not in all_results:
                all_results[eid] = _Result(intent_id, merged_slot_names)

    example_ids, utterances, labels, intents = load_tsv(eval_input)
    pred_intents = []
    label_intents = []
    for eid, intent in zip(example_ids, intents):
        label_intents.append(label2index(intent2idx, intent))
        pred_intents.append(all_results[eid].intent)
    intent_acc = sklearn.metrics.accuracy_score(label_intents, pred_intents)
    log.info("Intent Accuracy: %.4f" % intent_acc)

    pred_icsl = []
    label_icsl = []
    for eid, intent, slot_labels in zip(example_ids, intents, labels):
        label_icsl.append(str(label2index(intent2idx, intent)) + ' ' + ' '.join(slot_labels))
        pred_icsl.append(str(all_results[eid].intent) + ' ' + ' '.join(all_results[eid].slot_labels))
    exact_match = sklearn.metrics.accuracy_score(label_icsl, pred_icsl)
    log.info("Exact Match: %.4f" % exact_match)

    with open(conll_prediction_file, "w") as fw:
        for eid, utterance, labels in zip(example_ids, utterances, labels):
            preds = all_results[eid].slot_labels
            for w, l, p in zip(utterance, labels, preds):
                fw.write(' '.join([w, l, p]) + '\n')
            fw.write('\n')
    proc = subprocess.Popen(["perl", "conlleval.pl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    with open(conll_prediction_file) as f:
        stdout = proc.communicate(f.read().encode())[0]
    result = stdout.decode('utf-8').split('\n')[1]
    slot_f1 = float(result.split()[-1].strip())
    log.info("Slot Labeling: %s" % result)
    return intent_acc, slot_f1

# extract labels
train_input = data_dir + 'atis_train.tsv'
intent2idx, label2idx = get_label_indices(train_input)

for lang in ['EN', 'ES', 'DE', 'ZH', 'JA', 'PT', 'FR', 'HI', 'TR']:
    log.info('Train on %s:' % lang)
    model_name = 'model_bert_' + lang + '.' + str(random_seed)
    if lang == 'EN':
        train_input = data_dir + 'atis_train.tsv'
        dev_input = data_dir + 'atis_dev.tsv'
    else:
        train_input = data_dir + 'atis_train_' + lang + '.tsv'
        dev_input = data_dir + 'atis_dev_' + lang + '.tsv'
    train(model_name, train_input, dev_input)

log.info('==========Supervised learning==========')
for lang in ['EN', 'ES', 'DE', 'ZH', 'JA', 'PT', 'FR', 'HI', 'TR']:
    log.info('Evaluate on %s:' % lang)
    model_name = 'model_bert_' + lang + '.' + str(random_seed)
    if lang == 'EN':
        test_input = data_dir + 'atis_test.tsv'
    else:
        test_input = data_dir + 'atis_test_' + lang + '.tsv'
    evaluate(model_name=model_name, eval_input=test_input)

log.info('==========Transfer learning==========')
src_lang = 'EN'
for lang in ['EN', 'ES', 'DE', 'ZH', 'JA', 'PT', 'FR', 'HI', 'TR']:
    log.info('Evaluate on %s:' % lang)
    model_name = 'model_bert_' + src_lang + '.' + str(random_seed)
    if lang == 'EN':
        test_input = data_dir + 'atis_test.tsv'
    else:
        test_input = data_dir + 'atis_test_' + lang + '.tsv'
    evaluate(model_name=model_name, eval_input=test_input)
