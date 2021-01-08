import torch
import torch.nn as nn
from transformers import BertModel


class TextClassificationModel(nn.Module):
    def __init__(self, rnn_dim, rnn_num_layer, im_dim, que_dim, key_dim, deep, num_class):
        super(TextClassificationModel, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.rnn = nn.LSTM(input_size=768, hidden_size=rnn_dim, num_layers=rnn_num_layer, batch_first=True,
                           dropout=0.2, bidirectional=True)
        self.rnn_im = nn.Linear(rnn_dim*2, im_dim)
        self.im_out = nn.Linear(im_dim, num_class)

        # Att part
        self.pro_query = nn.Linear(que_dim, deep)
        self.pro_key = nn.Linear(key_dim, deep)
        self.pro_value = nn.Linear(rnn_dim * 2, rnn_dim * 2)

    def forward(self, inputs, sample_m, sentence_ms, sen_mask, using_GPU=False):

        rnn_re, _ = self.rnn(inputs)
        rnn_re = self.dropout(rnn_re)

        batch_query = self.pro_query(sample_m).unsqueeze(1)
        batch_keys = self.pro_key(sentence_ms).permute(0, 2, 1)
        batch_values = rnn_re
        batch_logits = torch.matmul(batch_query, batch_keys)
        batch_logits = batch_logits + sen_mask.unsqueeze(1)
        batch_weights = nn.functional.softmax(batch_logits, dim=-1)
        result = torch.matmul(batch_weights, batch_values).squeeze(1)

        im = self.rnn_im(result)
        im = self.dropout(im)
        out = self.im_out(im)
        output = nn.functional.log_softmax(out, dim=-1)
        return output

