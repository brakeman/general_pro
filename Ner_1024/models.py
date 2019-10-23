from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input, Sequential
from keras.layers import LSTM, Embedding, SpatialDropout1D, concatenate, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import json
import warnings
warnings.filterwarnings("ignore")


def get_cos_similarity(sms, templates):
    '''计算一条新sms 与 每个template 相似度;'''
    def cos_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b))
    return [cos_sim(i, sms) for i in templates]

def tfIdfVector(corpus):
    '''corpus is a list of sentences:
    ['This is an example', 'hello world', ...]
    '''
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    x = vectorizer.fit_transform(corpus)
    tfidf = transformer.fit_transform(x)
    return tfidf.toarray()


class IndiaSms:
    '''
    处理1条或多条短信类别及NER；
    1. 输入模型储存路径 和 参数储存路径，fit函数会重新训练model 并保存复现文件：model, para_dict
    例子：
    test = IndiaSms()
    test.fit(DF, batch_size=32, epochs=1, savePath = 'model_test_01.h5',
        validation_split=0.1, paraname='paradict_test_01')

    '''

    def __init__(self, model_path=None, para_dict=None):
        if model_path and para_dict:
            self.max_len = para_dict['max_len']
            self.max_len_char = para_dict['max_len_char']
            self.n_tags = para_dict['n_tags']
            self.char2idx = para_dict['char2idx']
            self.word2idx = para_dict['word2idx']
            self.tag2idx = para_dict['tag2idx']
            self.model = self._model_load(model_path)


    def fit(self, DF, batch_size, epochs,
            validation_split, paraname, savePath):
        '''DF like
           # token # lable # sent_id
           ############################
           # This  # other # send_id_0
           # is    # other # send_id_0
           # an    # other # send_id_0
           # apple # Item  # send_id_0
           # Pls   # other # send_id_1
           # pay   # other # send_id_1
           # RMB   # other # send_id_1
           # 100   # Money # send_id_1
        '''
        X_word, X_char, y, max_len, max_len_char, n_tags, word2idx, char2idx, tag2idx = self._preprocess_for_fit(DF)
        model = self.NN_model(word2idx, char2idx, max_len, max_len_char, n_tags)
        y_tr = [to_categorical(i, num_classes=n_tags) for i in y]

        history = model.fit([X_word, np.array(X_char).reshape((len(X_char), max_len, max_len_char))],
                            np.array(y_tr),
                            batch_size=batch_size, epochs=epochs, validation_split=validation_split)
        model.save(savePath)

        paras = {}
        paras['max_len'] = max_len
        paras['max_len_char'] = max_len_char
        paras['word2idx'] = word2idx
        paras['char2idx'] = char2idx
        paras['tag2idx'] = tag2idx
        paras['n_tags'] = n_tags

        with open(paraname + '.json', 'w') as outfile:
            json.dump(paras, outfile, ensure_ascii=False)
            outfile.write('\n')

        self.model = model
        self.max_len = max_len
        self.max_len_char = max_len_char
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.n_tags = n_tags
        return

    def predict(self, sms):
        '''
        sms type:
        [['this', 'is', 'an', 'example']]
        '''
        idx2tag = {i: w for w, i in self.tag2idx.items()}
        X_word = self._get_X_word(sms, self.word2idx, self.max_len)
        X_char = self._get_X_char(sms, self.char2idx, self.max_len, self.max_len_char)
        y_pred = self.model.predict([X_word, np.array(X_char).reshape((len(X_char), self.max_len, self.max_len_char))])
        p = np.argmax(y_pred, axis=-1)
        pred = [[idx2tag[i] for i in sent] for sent in p]
        seq_len = [len(i) for i in sms]
        return [pre[:len_] for pre, len_ in zip(pred, seq_len)]

    def NN_model(self, word2idx, char2idx,
                 max_len, max_len_char, n_tags):
        word_in = Input(shape=(max_len,))
        n_words = len(word2idx.keys())
        n_chars = len(char2idx.keys())
        emb_word = Embedding(input_dim=n_words, output_dim=30,
                             input_length=max_len, mask_zero=True)(word_in)
        char_in = Input(shape=(max_len, max_len_char,))
        emb_char = TimeDistributed(Embedding(input_dim=n_chars, output_dim=10,
                                             input_length=max_len_char, mask_zero=True))(char_in)

        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.3))(emb_char)
        x = concatenate([emb_word, char_enc])
        x = SpatialDropout1D(0.3)(x)
        main_lstm = Bidirectional(LSTM(units=30, return_sequences=True,
                                       recurrent_dropout=0.4))(x)
        model = TimeDistributed(Dense(35, activation='relu'))(main_lstm)
        crf = CRF(n_tags, learn_mode='marginal')
        out = crf(model)  # prob
        model = Model([word_in, char_in], out)
        #         from keras.utils.vis_utils import plot_model
        #         plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        model.compile(optimizer='rmsprop',
                      loss=crf_loss,
                      metrics=[crf.accuracy])
        # model.summary()
        return model

    def _model_load(self, path):
        model = self.NN_model(self.word2idx, self.char2idx,
                              self.max_len, self.max_len_char, self.n_tags)
        model.load_weights(path)
        return model

    def _preprocess_for_fit(self, DF):
        '''DF like
           # token # lable # sent_id
           ############################
           # This  # other # send_id_0
           # is    # other # send_id_0
           # an    # other # send_id_0
           # apple # Item  # send_id_0
           # Pls   # other # send_id_1
           # pay   # other # send_id_1
           # RMB   # other # send_id_1
           # 100   # Money # send_id_1
        '''
        max_len = 75
        max_len_char = 13
        words = list(set(DF['token'].values))
        words = list(set([i.lower() for i in words]))
        tags = list(set(DF['lable'].values))
        n_tags = len(tags)
        grouped = DF.groupby('sent_id').apply(
            lambda x: [(w.lower(), t) for w, t in zip(x.token.tolist(), x.lable.tolist())])
        sentences = [s for s in grouped]

        word2idx = {w: i + 2 for i, w in enumerate(words)}
        word2idx['UNK'] = 1
        word2idx['PAD'] = 0

        tag2idx = {t: i for i, t in enumerate(tags)}

        chars = set([w_i for w in words for w_i in w])
        char2idx = {c: i + 2 for i, c in enumerate(chars)}
        char2idx['UNK'] = 1
        char2idx['PAD'] = 0

        # 句子词序列 转为 句子词序号序列
        sentences_X = [[tuple_[0] for tuple_ in sent] for sent in sentences]
        X_word = self._get_X_word(sentences_X, word2idx, max_len)
        X_char = self._get_X_char(sentences_X, char2idx, max_len, max_len_char)
        y = [[tag2idx[w[1]] for w in s] for s in sentences]
        y = pad_sequences(maxlen=max_len, sequences=y,
                          value=tag2idx['other'], padding='post', truncating='post')
        return X_word, X_char, y, max_len, max_len_char, n_tags, word2idx, char2idx, tag2idx

    def _get_X_word(self, sentences, word2idx, max_len):
        '''sentences type: [['this','is','an','example']]'''
        X_word = [[word2idx.get(w[0], word2idx['UNK']) for w in s] for s in sentences]
        X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx['PAD'], padding='post',
                               truncating='post')
        return X_word

    def _get_X_char(self, sentences, char2idx, max_len, max_len_char):
        '''sentences type: [['this','is','an','example']]'''
        X_char = []
        for sentence in sentences:
            sent_seq = []
            for i in range(max_len):
                word_seq = []
                for j in range(max_len_char):
                    try:
                        word_seq.append(char2idx.get(sentence[i][j]))
                    except:
                        word_seq.append(char2idx.get('PAD'))
                sent_seq.append(word_seq)
            X_char.append(np.array(sent_seq))
        return X_char


class SmsRuleClf:
    '''模块功能: 给定一个(组)sms， 能够对其正确分类'''

    def __init__(self, labeled_templates_df):
        self.labeled_templates_df = labeled_templates_df
        self.corpus, self.labels = self._get_template_corpus_labels()

    def _get_template_corpus_labels(self):
        corpus, labels = self.labeled_templates_df.sms.tolist(), self.labeled_templates_df.label.tolist()
        return corpus, labels

    def predict(self, sms):
        '''
        Input: ['This is an example', 'hello world', ...]
        Output: [cls1, cls2, ...]
        '''
        if isinstance(sms, list) and isinstance(sms[0], str):
            template_corpus, template_labels = self.corpus, self.labels
            num_sms = len(sms)
            for single in sms:
                template_corpus.append(single.lower())
            all_tfidf = tfIdfVector(template_corpus)
            template_tfidf = all_tfidf[:-num_sms]
            instances_tfidf = all_tfidf[-num_sms:]

            result = []
            for idx, single_sms in enumerate(instances_tfidf):
                cos_score = get_cos_similarity(single_sms, template_tfidf)
                max_score = np.max(cos_score)
                label = template_labels[np.argmax(cos_score)]
                result.append([sms[idx], label, max_score])
            return result
        else:
            raise Exception('''sms type not allowed: should be with type: ['This is an example', 'hello world', ...]''')


def generate_clf_ner(sms, gsm_templates_df, model, para_dict, return_df=True):
    '''
    sms example: ['successful transfer of rs 1800 to shubham pa using airtel money transfer by - 919892750965.txn id: 1170827753.charges - max 0.65%',
                  'thank you for using your citibank debit card 5497xxxxxxxx2902 for rs. 6325 at sbicard-billdesk on 28-sep-18.']

    '''
    sms = [i.lower() for i in sms]
    sms_ner = [i.split() for i in sms]

    # CLF
    clf = SmsRuleClf(gsm_templates_df)
    clf_result = clf.predict(sms)

    # NER
    Ner = IndiaSms(model, para_dict=para_dict)
    ner_result = Ner.predict(sms_ner)

    # combine
    record = []
    result_list = []
    for clf_, ner_ in zip(clf_result, ner_result):
        result_dict = {}
        ner_res = [{i: clf_[0].split()[j]} for j, i in enumerate(ner_) if i != 'other']
        record.append([clf_[0], clf_[1], clf_[2], ner_res])
        result_dict['sms'] = clf_[0]
        result_dict['class'] = clf_[1]
        result_dict['class_score'] = clf_[2]
        result_dict['ner_result'] = ner_res
        result_list.append(result_dict)
        
    if return_df:
        Final_DF = pd.DataFrame.from_records(record)
        Final_DF.columns = ['sms', 'class', 'class_score', 'ner_result']
        return Final_DF
    else:
        return result_list
