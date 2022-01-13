### Environment
```
pip install numpy scipy scikit-learn
pip install --upgrade mxnet>=1.6.0
pip install gluonnlp
```

### Data
+ Multilingual ATIS dataset in EN, ES, DE, ZH, JA, PT, FR, HI, and TR.

### Preparation
+ Download BERT related [codes](https://gluon-nlp.mxnet.io/model_zoo/bert/index.html).
+ Decompress it and place the folder in `code`.
+ Install [fast-align](https://github.com/clab/fast_align).

### Run
+ For supervised experiments, run `python lstm_alone.py $seed` or `python bert_alone.py $seed` to train the biLSTM/BERT supervised model (`$seed` is a random seed number).
+ For multilingual experiments, run `python lstm_joint.py $seed` or `python bert_joint.py $seed` to train the biLSTM/BERT multilingual model.
+ For cross-lingual transfer using *MT+fast-align*, first run `python translate_and_align.py $lang` to translate the English utterances to the target language `$lang` and project the slot labels using fast-align. And then, run `python lstm_mt.py $seed` or `python bert_mt.py $seed` to train the biLSTM/BERT model.
+ For cross-lingual transfer using *MT+soft-align*, first run `python translate_and_align.py $lang` to translate the English utterances to the target language `$lang`. And then, run `python bert_soft_align.py $seed` to train the soft-alignment model.
