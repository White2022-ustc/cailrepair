''' 
_*_ coding: utf-8 _*_
Date: 2020/6/26
Author: YZL
Intent: MCQA robert model
'''

from transformers import BertModel, BertLayer
import torch.nn as nn
import torch


class FirstModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['pretrain_model_dir'])  # pretrain_model_dir2
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, sample):
        inputs_ids1, inputs_masks1 = sample['inputs_ids1'], sample['inputs_masks1']
        types_ids1 = sample['type_ids1']
        inputs_ids2, inputs_masks2 = sample['inputs_ids2'], sample['inputs_masks2']
        types_ids2 = sample['type_ids2']
        inputs_ids3, inputs_masks3 = sample['inputs_ids3'], sample['inputs_masks3']
        types_ids3 = sample['type_ids3']
        cls1 = self.bert(inputs_ids1, attention_mask=inputs_masks1, token_type_ids=types_ids1)[1]
        logits1 = self.linear(cls1)
        score1 = torch.sigmoid(logits1).squeeze(-1)
        cls2 = self.bert(inputs_ids2, attention_mask=inputs_masks2, token_type_ids=types_ids2)[1]
        logits2 = self.linear(cls2)
        score2 = torch.sigmoid(logits2).squeeze(-1)
        cls3 = self.bert(inputs_ids3, attention_mask=inputs_masks3, token_type_ids=types_ids3)[1]
        logits3 = self.linear(cls3)
        score3 = torch.sigmoid(logits3).squeeze(-1)
        score = 0.5*score1+0.3*score2+0.2*score3
        print(score)
        return score


if __name__ == "__main__":
    from lajs_config import LajsConfig
    from lajs_datasets import SegmentSelectionDataset, select_collate
    from torch.utils.data import DataLoader
    d = SegmentSelectionDataset(LajsConfig, LajsConfig['train_file'])
    loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=2, collate_fn=select_collate)
    model = FirstModel(LajsConfig)
    for i, item in enumerate(loader):
        #print(item)
       #print('*' * 20)
       # print(model(item))
        if i > 4:
            break

    pass