import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import CrossEntropyLoss


class TextClassificationModel(nn.Module):
    def __init__(self, embedding_dim, cnn_dim, que_dim, key_dim, deep, num_class, using_RNN, rnn_dim, rnn_num_layer,
                 bidr=True):
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
        self.cnn1 = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_dim, kernel_size=2)
        self.cnn2 = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_dim, kernel_size=3)
        # max_pooling不放在这里，不需要学习 但注意使用时需要注意是否放在gpu上
        self.dropout_cnn = nn.Dropout(0.2)

        # Att part
        self.pro_query = nn.Linear(que_dim, deep)
        self.pro_key = nn.Linear(key_dim, deep)
        self.pro_value = nn.Linear(rnn_dim*2, rnn_dim*2)

        # RNN part
        self.using_RNN = False
        if using_RNN:
            self.using_RNN = True
            self.rnn = nn.LSTM(input_size=cnn_dim*2, hidden_size=rnn_dim, num_layers=rnn_num_layer, batch_first=True,
                               dropout=0.2, bidirectional=bidr)
            self.bidc = 2 if bidr else 1
            # self.pro_value = nn.Linear(rnn_dim*bidc, rnn_dim*bidc)

        self.out_dropout = nn.Dropout(0.2)
        self.out_label = nn.Linear(cnn_dim*2, num_class)
        if using_RNN:
            self.out_label = nn.Linear(rnn_dim*self.bidc, num_class)
            self.rnn_mlp = nn.Linear(rnn_dim*self.bidc+key_dim, rnn_dim*self.bidc)

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

        cnn_re1 = self.cnn1(inputs.view(-1, sen_len, embedding_dim).permute(0, 2, 1))
        cnn_re2 = self.cnn2(inputs.view(-1, sen_len, embedding_dim).permute(0, 2, 1))
        # cnn_re: shape(batch_size*sen_num, cnn_dim, cnn_len)
        # print('cnn_re', cnn_re1.size())

        max_p_m1 = nn.AvgPool1d(cnn_re1.size(-1)).cuda() if using_GPU else nn.AvgPool1d(cnn_re1.size(-1))
        max_p_m2 = nn.AvgPool1d(cnn_re2.size(-1)).cuda() if using_GPU else nn.AvgPool1d(cnn_re2.size(-1))
        mp_re1 = max_p_m1(cnn_re1).squeeze(-1).view(batch_size, sen_num, -1)
        mp_re2 = max_p_m2(cnn_re1).squeeze(-1).view(batch_size, sen_num, -1)
        # mp_re: shape(batch_size, sen_num, cnn_dim)
        # print('mp_re', mp_re1.size())

        temp_re = torch.cat([mp_re1, mp_re2], dim=-1)
        # temp_re: shape(batch_size, sen_num, cnn_dim*2)
        # print('temp_re', temp_re.size())
        temp_re = self.dropout_cnn(temp_re)
        cnn_re = temp_re

        if self.using_RNN:
            temp_re, _ = self.rnn(temp_re)
            # temp_re: shape(batch_size, sen_num, rnn_dim*2)
            # print('temp_re', temp_re.size())
            temp_re = self.dropout_cnn(temp_re)

            # 使用残差与Norm结构请保证cnn_dim=rnn_dim
            temp_re = temp_re + cnn_re
            temp_re = temp_re.permute(1, 0, 2)
            temp_re = torch.layer_norm(temp_re, temp_re.size()[1:])
            temp_re = temp_re.permute(1, 0, 2)

        batch_query = self.pro_query(sample_m).unsqueeze(1)
        # batch_query: shape(batch_size, 1, deep)
        # print('batch_query', batch_query.size())
        batch_query = torch.tanh(batch_query)

        batch_keys = self.pro_key(sentence_ms).permute(0, 2, 1)
        # batch_keys: shape(batch_size, deep, sen_num)
        # print('batch_keys', batch_keys.size())
        batch_keys = torch.tanh(batch_keys)

        # batch_values = temp_re
        # batch_values = self.pro_value(temp_re)
        batch_values = torch.cat([self.pro_value(temp_re), sentence_ms], dim=-1)
        batch_values = self.rnn_mlp(batch_values)

        batch_logits = torch.matmul(batch_query, batch_keys)
        # batch_logits = torch.nn.functional.layer_norm(batch_logits, normalized_shape=batch_logits.size()[:])
        batch_logits = batch_logits + sen_mask.unsqueeze(1)

        batch_weights = nn.functional.softmax(batch_logits, dim=-1)
        # batch_weights: shape(batch_size, 1, sen_num)
        # print('batch_weights', batch_weights.size())

        result = torch.matmul(batch_weights, batch_values).squeeze(1)
        # result: shape(batch_size, rnn/cnn_dim)
        # print('result:', result.size())

        # 不使用att
        # std = int(temp_re.size(-1)/2)
        # result = torch.cat([temp_re[:, -1, :std], temp_re[:, 0, std:]], dim=-1).squeeze(1)

        result = self.out_dropout(result)
        out = self.out_label(result)
        # out: shape(batch_size, num_class)
        output = nn.functional.log_softmax(out, dim=-1)
        return output


class BertFTC(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="bert-base-uncased",
    #     output_type=TokenClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            class_weights = torch.FloatTensor([1, 1])  # 根据需求修改 包括cuda 虽然是在模型里面 但当有中间变量出现时需要注意cuda
            loss_fct = CrossEntropyLoss(weight=class_weights)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
