import pandas as pd
import random

def gen_new(ori_str):
    return ''.join([str(random.randint(0,9)) if i.isdigit() else i for i in list(ori_str)])

def create_fake_label(DF, fake_times):
    DF2 = DF.copy()
    for i in range(fake_times):
        col = DF2.apply(lambda x: gen_new(x.token) if x.lable!='other' else x.token, axis=1)
        DF2['new_tokens_{}'.format(i)] = col
    return DF2

def generate_sent_id(sent_id_list, origin_sent_id):
    dic = {}
    for k in origin_sent_id:
        tmp = 'send_id_' + str(sent_id_list[-1])
        sent_id_list.pop()
        dic.setdefault(k, tmp)
    return sent_id_list, dic

def fake2real(df, fake_times):
    lis = []
    sent_id_list = list(range(len(df.sent_id.unique()) * fake_times))
    origin_sent_id = df.sent_id.unique()
    for i in range(fake_times):
        tmp_df = df[['new_tokens_{}'.format(i), 'lable', 'sent_id']]
        tmp_df.columns = ['token', 'lable', 'sent_id']
        sent_id_list, dic = generate_sent_id(sent_id_list, origin_sent_id)
        tmp_df.sent_id = tmp_df.sent_id.map(dic)
        lis.append(tmp_df)
    return pd.concat(lis)
