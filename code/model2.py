import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import CrossEntropyLoss


class TextClassificationModel(nn.Module):
    def __init__(self, embedding_dim, que_dim, key_dim, deep, num_class, rnn_dim, rnn_num_layer, im_dim):
        """
        可调参数：cnn核大小；是否使用rnn，使用几层；values是否经过变换；是否使用多头att；
                是否添加sentence pos embedding，加在哪里；残差模块
        :param embedding_dim: (int) dim of token
        :param cnn_dim: (int) cnn filters numbers
        :param que_dim: (int) sample metaphor feature dim
        :param key_dim: (int) sentence metaphor feature dim
        :param deep: (int) metaphor covert dim
        :param num_class: (int) number of classes
        :param using_RNN: (bool)
        :param rnn_dim: (int) hidden_size of rnn
        :param rnn_num_layer: (int) num of layers in rnn
        :param bidr: (bool) bidirectional rnn or not
        """
        super(TextClassificationModel, self).__init__()
        self.rnn_dim = rnn_dim

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_dim, num_layers=rnn_num_layer, batch_first=True,
                           dropout=0.2, bidirectional=True)
        self.rnn_dropout = nn.Dropout(0.2)
        self.rnn_im = nn.Linear(rnn_dim*2, im_dim)
        self.im_rnn = nn.Linear(im_dim, rnn_dim*2)

        self.att_im = nn.Linear(rnn_dim*2, im_dim)
        self.im_re = nn.Linear(im_dim, rnn_dim * 2)

        # Att part
        self.pro_query = nn.Linear(que_dim, deep)
        self.pro_key = nn.Linear(key_dim, deep)
        self.pro_value = nn.Linear(rnn_dim*2, rnn_dim*2)

        self.out_label = nn.Linear(rnn_dim*2, num_class)
        self.out_dropout = nn.Dropout(0.2)

    def forward(self, inputs, sample_m, sentence_ms, sen_mask, using_GPU=False):
        """
        为了方便 输入的数据使用pad过的，每个batch下sentence_num和sentence_len保持一致
        输入数据全部为tensor数据
        :param sen_mask: mask the padding sentence(int(0)). shape(batch_size, sentence_num)
        :param inputs: shape(batch_size, sentence_num, sentence_len, embedding_dim)
        :param sample_m: shape(batch_size, metaphor_dim)
        :param sentence_ms: shape(batch_size, sentence_num, metaphor_dim)
        :return: label distribution shape(batch_size, num_class)
        """
        batch_size, sen_num, sen_len, embedding_dim = inputs.size()

        rnn_re, _ = self.rnn(inputs.view(-1, sen_len, embedding_dim))
        temp_re = torch.cat([rnn_re[:, -1, :self.rnn_dim], rnn_re[:, 0, self.rnn_dim:]], dim=-1).view(batch_size,
                                                                                                      sen_num, -1)
        temp_re = self.rnn_dropout(temp_re)
        temp_re = self.rnn_im(temp_re)
        temp_re = self.im_rnn(temp_re)

        batch_query = self.pro_query(sample_m).unsqueeze(1)
        batch_query = torch.tanh(batch_query)

        batch_keys = self.pro_key(sentence_ms).permute(0, 2, 1)
        batch_keys = torch.tanh(batch_keys)

        batch_values = self.pro_value(temp_re)

        batch_logits = torch.matmul(batch_query, batch_keys)
        batch_logits = batch_logits + sen_mask.unsqueeze(1)

        batch_weights = nn.functional.softmax(batch_logits, dim=-1)

        result = torch.matmul(batch_weights, batch_values).squeeze(1)
        result = self.att_im(result)
        result = self.im_re(result)

        result = self.out_dropout(result)
        out = self.out_label(result)
        # out: shape(batch_size, num_class)
        output = nn.functional.log_softmax(out, dim=-1)
        return output

