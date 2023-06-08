'''
_*_ coding: utf-8 _*_
Date: 2020/6/15
Author: YZL
Intent: creat datasets for pytorch
'''


import json
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random



# ''' doc segment selection'''
# class SegmentSelectionDataset(object):
#     def __init__(self, config, data_file=None):
#         self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_dir'])
#         self.chunk_size = config['chunk_size']
#         self.config = config

#         if data_file is not None:
#             with open(data_file, 'r', encoding='utf8') as f:
#                 self.data = json.load(f)
#         else:
#             self.data = None

#     def chunk(self, text):  ### 窗口不重叠
#         chunks_tokens = []
#         tokens = self.tokenizer.tokenize(text)
#         num = len(tokens) // self.chunk_size
#         if num * self.chunk_size < len(tokens):
#             num += 1
#         for i in range(num):
#             chunks_tokens.append(tokens[i * self.chunk_size:(i + 1) * self.chunk_size])
#         return chunks_tokens


#     def chunk_overlap(self, text, window_size=254, overlap=127, cutoff=True):  ### 窗口重叠  64
#         chunks_tokens = []
#         tokens = self.tokenizer.tokenize(text)
#         start = 0
#         while start < len(tokens):
#             window_tokens = tokens[start:start+window_size]
#             if cutoff:
#                 if len(window_tokens) > 20:
#                     chunks_tokens.append(window_tokens)
#             else:
#                 chunks_tokens.append(window_tokens)
#             start += overlap
#         return chunks_tokens

#     def pad_seq(self, ids_list, types_list):
#         batch_len = 512  # max([len(ids) for ids in ids_list])
#         new_ids_list, new_types_list, new_masks_list = [], [], []
#         for ids, types in zip(ids_list, types_list):
#             masks = [1] * len(ids) + [0] * (batch_len - len(ids))
#             types += [0] * (batch_len - len(ids))
#             ids += [0] * (batch_len - len(ids))
#             new_ids_list.append(ids)
#             new_types_list.append(types)
#             new_masks_list.append(masks)
#         return new_ids_list, new_types_list, new_masks_list

#     def encode(self, q_crime, d_crime, q, d, q_chunksize=100, d_chunksize=100, cutoff=True):  # q_chunksize=3, d_chunksize=15 for train
#         ids, type_ids = [], []
#         q_crime_tokens, d_crime_tokens = self.tokenizer.tokenize(q_crime), self.tokenizer.tokenize(d_crime)
#         crime_tokens = ['[CLS]'] + q_crime_tokens + ['[SEP]'] + d_crime_tokens + ['[SEP]']
#         crime_ids = self.tokenizer.convert_tokens_to_ids(crime_tokens)
#         crime_types = [0] * (len(q_crime_tokens) + 2) + [1] * (len(d_crime_tokens) + 1)
#         ids.append(crime_ids)
#         type_ids.append(crime_types)

#         q_chunks, d_chunks = self.chunk(q), self.chunk_overlap(d, cutoff=cutoff)
#         q_chunks, d_chunks = q_chunks[:q_chunksize], d_chunks[:d_chunksize]
#         for q_chunk in q_chunks:
#             for d_chunk in d_chunks:
#                 tokens = ['[CLS]'] + q_chunk + ['[SEP]'] + d_chunk + ['[SEP]']
#                 token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#                 types = [0] * (len(q_chunk) + 2) + [1] * (len(d_chunk) + 1)
#                 ids.append(token_ids)
#                 type_ids.append(types)
#         ids, type_ids, masks = self.pad_seq(ids, type_ids)
#         return ids, type_ids, masks, len(q_chunks), len(d_chunks)

#     def __getitem__(self, index):
#         data = self.data[index]
#         qidx, q, q_crime, docs = data['qidx'], data['q'], data['crime'], data['docs']
#         labels = data['labels']

#         pos_docs, neg_docs = [], []
#         for d in docs:
#             if d['didx'] in labels:
#                 if labels[d['didx']] > 1:
#                     pos_docs.append(d)
#                 elif labels[d['didx']] <= 1:
#                     neg_docs.append(d)
#             else:
#                 neg_docs.append(d)
#         sample_docs = pos_docs + random.sample(neg_docs,
#                                                k=len(pos_docs) if 2 * len(pos_docs) >= self.config['random_k'] else
#                                                self.config['random_k'] - len(pos_docs))
#         docs = random.sample(sample_docs, k=self.config['random_k'])

#         rel_labels = []
#         select_token_ids, select_type_ids, select_token_masks = [], [], []
#         for doc in docs:
#             didx, d_crime, content, max_score_indexes = doc['didx'], doc['d_crime'], doc['content'], doc['max_indexes']
#             token_ids, type_ids, token_masks, num_q, num_d = self.encode(q_crime, d_crime, q, content)
#             try:
#                 max_score_indexes = [0] + [i * num_q + index[0] + 1 for i, index in enumerate(max_score_indexes) if len(index) > 0]
#             except:
#                 print('**********', max_score_indexes, content, token_ids, '**********')
#                 exit()
#             for j, t_ids in enumerate(token_ids):
#                 if j in max_score_indexes:
#                     select_token_ids.append(t_ids)
#                     select_token_masks.append(token_masks[j])
#                     select_type_ids.append(type_ids[j])
#                     rel_labels.append(labels[didx] / 3 if didx in labels else 0)

#         return {'inputs_ids': torch.LongTensor(select_token_ids),
#                 'type_ids': torch.LongTensor(select_type_ids),
#                 'inputs_masks': torch.LongTensor(select_token_masks),
#                 'rel_labels': torch.Tensor(rel_labels)}

#     def get_predict_samples(self, q, q_crime, d):
#         d_crime, content = d['d_crime'], d['content']
#         token_ids, type_ids, token_masks, num_q, num_d = self.encode(q_crime, d_crime, q, content, cutoff=False)
#         return {'inputs_ids': torch.LongTensor(token_ids),
#                 'type_ids': torch.LongTensor(type_ids),
#                 'inputs_masks': torch.LongTensor(token_masks),
#                 'num_q': num_q,
#                 'num_d': num_d}

#     def __len__(self):
#         return len(self.data)


def read_file(query_path):
    with open(query_path,'r',encoding='utf-8') as file:
        data=[]
        for line in file:
            a=json.loads(line)
            data.append(a)
    return data
def read_ford(input_folder,a):
    input_filename = str(a) + '.json'
    input_file = os.path.join(input_folder,input_filename)
    data = []
    if os.path.isfile(input_file):
        with open(input_file,'r',encoding='utf-8') as f:
            data = json.load(f)
    return data
class SegmentSelectionDataset(object):
    def __init__(self, config, data_file=None):
        self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_dir'])
        self.chunk_size = config['chunk_size']
        self.config = config
        self.query = read_file("/home/jshi/c/ccc/query.json")


    def chunk(self, text):  ### 窗口不重叠
        chunks_tokens = []
        tokens = self.tokenizer.tokenize(text)
        num = len(tokens) // self.chunk_size
        if num * self.chunk_size < len(tokens):
            num += 1
        for i in range(num):
            chunks_tokens.append(tokens[i * self.chunk_size:(i + 1) * self.chunk_size])
        return chunks_tokens

    def pad_seq(self, ids_list, types_list):
        batch_len = 512  # max([len(ids) for ids in ids_list])
        new_ids_list, new_types_list, new_masks_list = [], [], []
        for ids, types in zip(ids_list, types_list):
            masks = [1] * len(ids) + [0] * (batch_len - len(ids))
            types += [0] * (batch_len - len(ids))
            ids += [0] * (batch_len - len(ids))
            new_ids_list.append(ids)
            new_types_list.append(types)
            new_masks_list.append(masks)
        return new_ids_list, new_types_list, new_masks_list

    def encode(self,  q, d):  # q_chunksize=3, d_chunksize=15 for train
        ids, type_ids = [], []
        tokens = ['[CLS]'] + q + ['[SEP]'] + d + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(q) + 2) + [1] * (len(d) + 1)
        ids, type_ids, masks = self.pad_seq([token_ids], [types])
        #TODO 多分段处理
        return ids[0], type_ids[0], masks[0]

    def __getitem__(self,c):

        for j in range(1,10):
            self.docs = read_ford("/home/jshi/c/ccc/KnowledgeGraphCourse-master/doc_poc",j)
            self.docs_neg = read_ford("/home/jshi/c/ccc/KnowledgeGraphCourse-master/doc_neg",j)
            self.docs_poc = read_ford("/home/jshi/c/ccc/KnowledgeGraphCourse-master/doc",j)
            query=self.query[j]

            q_s25, q_s26, q_s23 = query['s25'].replace("\n", ""), query['s26'].replace("\n", ""), query['s23'].replace("\n", "")
            q_s25_chunk=self.chunk(q_s25)[0]
            q_s26_chunk=self.chunk(q_s26)[0]
            q_s23_chunk=self.chunk(q_s23)[0]
            labels = []
            select_token_ids, select_type_ids, select_token_masks = [], [], []
            for i in range(0,10):
                docs=self.docs[i]
                d_s25, d_s26, d_s23 = docs['s25'].replace("\n", ""), docs['s26'].replace("\n", ""), docs['s23'].replace("\n", "")
                try:
                    d_s25_chunk=self.chunk(d_s25)[0]
                    d_s26_chunk=self.chunk(d_s26)[0]
                    d_s23_chunk=self.chunk(d_s23)[0]
                    token_ids, type_ids, token_masks=self.encode(q_s25_chunk,d_s25_chunk)
                    select_token_ids.append(token_ids)
                    select_type_ids.append(type_ids)
                    select_token_masks.append(token_masks)
                    labels.append(1)
                    token_ids, type_ids, token_masks=self.encode(q_s26_chunk,d_s26_chunk)
                    select_token_ids.append(token_ids)
                    select_type_ids.append(type_ids)
                    select_token_masks.append(token_masks)

                    token_ids, type_ids, token_masks=self.encode(q_s23_chunk,d_s23_chunk)
                    select_token_ids.append(token_ids)
                    select_type_ids.append(type_ids)
                    select_token_masks.append(token_masks)

                except:
                    print('error!')

            for i in range(0,5):

                docs=self.docs_neg[i]
                d_s25, d_s26, d_s23 = docs['s25'].replace("\n", ""), docs['s26'].replace("\n", ""), docs['s23'].replace("\n", "")
                d_s25_chunk=self.chunk(d_s25)[0]
                d_s26_chunk=self.chunk(d_s26)[0]
                d_s23_chunk=self.chunk(d_s23)[0]
                token_ids, type_ids, token_masks=self.encode(q_s25_chunk,d_s25_chunk)
                select_token_ids.append(token_ids)
                select_type_ids.append(type_ids)
                select_token_masks.append(token_masks)
                labels.append(0.667)
                token_ids, type_ids, token_masks=self.encode(q_s26_chunk,d_s26_chunk)
                select_token_ids.append(token_ids)
                select_type_ids.append(type_ids)
                select_token_masks.append(token_masks)

                token_ids, type_ids, token_masks=self.encode(q_s23_chunk,d_s23_chunk)
                select_token_ids.append(token_ids)
                select_type_ids.append(type_ids)
                select_token_masks.append(token_masks)


            for i in range(0, 5):

                docs = self.docs_poc[ i]
                d_s25, d_s26, d_s23 = docs['s25'].replace("\n", ""), docs['s26'].replace("\n", ""), docs['s23'].replace(
                    "\n", "")
                d_s25_chunk = self.chunk(d_s25)[0]
                d_s26_chunk = self.chunk(d_s26)[0]
                d_s23_chunk = self.chunk(d_s23)[0]
                token_ids, type_ids, token_masks = self.encode(q_s25_chunk, d_s25_chunk)
                select_token_ids.append(token_ids)
                select_type_ids.append(type_ids)
                select_token_masks.append(token_masks)
                labels.append(0)
                token_ids, type_ids, token_masks = self.encode(q_s26_chunk, d_s26_chunk)
                select_token_ids.append(token_ids)
                select_type_ids.append(type_ids)
                select_token_masks.append(token_masks)

                token_ids, type_ids, token_masks = self.encode(q_s23_chunk, d_s23_chunk)
                select_token_ids.append(token_ids)
                select_type_ids.append(type_ids)
                select_token_masks.append(token_masks)



        return {'inputs_ids': torch.LongTensor(select_token_ids),
                'type_ids': torch.LongTensor(select_type_ids),
                'inputs_masks': torch.LongTensor(select_token_masks),
                'rel_labels': torch.Tensor(labels)}
    def __len__(self):
        return len(self.query)

def select_collate(batch):
    inputs_ids, inputs_masks, types_ids, rel_labels = None, None, None, None
    for i, s in enumerate(batch):
        if i == 0:
            inputs_ids, inputs_masks, types_ids, rel_labels = s['inputs_ids'], s['inputs_masks'], s['type_ids'], s['rel_labels']
        else:
            inputs_ids = torch.cat([inputs_ids, s['inputs_ids']], dim=0)
            inputs_masks = torch.cat([inputs_masks, s['inputs_masks']], dim=0)
            types_ids = torch.cat([types_ids, s['type_ids']], dim=0)
            rel_labels = torch.cat([rel_labels, s['rel_labels']], dim=0)
    return {'inputs_ids': inputs_ids,
            'inputs_masks': inputs_masks,
            'type_ids': types_ids,
            'rel_labels': rel_labels}


def test_dataloader():
    from lajs_config import LajsConfig

    da = SegmentSelectionDataset(LajsConfig, LajsConfig['train_file'])
    loader = DataLoader(da, batch_size=2, shuffle=False, num_workers=2, collate_fn=select_collate)
    for i, item in enumerate(loader):
        # print(item)
        print(item['inputs_ids'].shape,item['inputs_masks'].shape,item['type_ids'].shape,item["rel_labels"].shape)
        if i > 5:
            break


if __name__ == "__main__":
    test_dataloader()
    pass
