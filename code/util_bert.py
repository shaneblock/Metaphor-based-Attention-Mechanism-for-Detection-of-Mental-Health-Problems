import torch
import csv
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np


def get_data(data_path):
    """
    获取已处理数据
    :param data_path: optional('train_an', 'test_an', 'train_de', 'test_de')
    :return:
        samples -> List[tuple(sample_text, label)]
        vocab
        sam_ms -> Tensor, shape(sam_num, metaphor_vector_dim)
        sam_sen_ms -> List[Tensor], Tensor shape(sen_num, metaphor_vector_dim)
    """
    sam_ms = []
    sam_sen_ms = []
    samples = []
    vocab = []
    with open('metaphor_re/'+data_path+'/sam_label_meta.csv', 'r', encoding='utf-8') as f:
        for raw in csv.reader(f):
            sam_id = raw[0]
            label = int(raw[1])
            sam_meta = [float(i) for i in raw[2:]]
            sam_ms.append(sam_meta)

            sam_text = []
            with open('metaphor_re/'+data_path+'/sens/'+str(sam_id)+'_sen.csv', 'r', encoding='utf-8') as f_sens:
                for sen in csv.reader(f_sens):
                    sam_text.append(sen)
                    vocab.extend(sen)
            samples.append((sam_text, label))

            sen_ms = []
            with open('metaphor_re/'+data_path+'/metaphor/'+str(sam_id)+'_meta.csv', 'r', encoding='utf-8') as f_ms:
                for sen_m in csv.reader(f_ms):
                    sen_ms.append([float(i) for i in sen_m])
            sam_sen_ms.append(torch.tensor(sen_ms))
            print('sample num: ', sam_id)
            assert len(sam_text) == len(sen_ms)
    vocab = set(vocab)
    return samples, vocab, torch.tensor(sam_ms), sam_sen_ms


def index_sentences(data):
    """
    对已分词数据进行bert索引
    :param data: List[samples], sample -> tuple(List[sentences], label)
    :return:
        indexed data, List[indexed_text] and List[label], embedded_text -> List[sentences], sentence -> List[tensor.float]
    """
    t = BertTokenizer.from_pretrained('bert-base-uncased')

    index_data = []
    labels = []
    for sample in data:
        index_sample = []
        for sentence in sample[0]:
            index_sen = t.convert_tokens_to_ids(sentence)
            index_sample.append(torch.tensor(index_sen, dtype=torch.long))
        labels.append(sample[1])
        index_data.append(index_sample)
    return index_data, labels


class TextDataset(Dataset):
    def __init__(self, embedded_text, sample_ms, sen_ms, labels):
        """

        :param embedded_text: shape(sample_num, sentence_num, sen_length, embedding_dim)
        :param sample_ms: shape(sample_num, sample_metaphor)
        :param sen_ms: shape(sample_num, sentence_num, sentence_metaphor)
        :param labels: shape(sample_num)
        """
        assert len(embedded_text) == len(labels)
        assert len(sample_ms) == len(labels)
        assert len(sen_ms) == len(labels)
        self.embedded_text = embedded_text
        self.labels = labels
        self.sample_ms = sample_ms
        self.sen_ms = sen_ms

    def __getitem__(self, idx):
        sample_text = self.embedded_text[idx]
        sample_m = self.sample_ms[idx]
        sen_m = self.sen_ms[idx]
        sample_label = self.labels[idx]
        return sample_text, sample_m, sen_m, sample_label

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        """
        填充数据 按bert能驾驭的最大数据尺寸处理 sample_text.shape = (batch_size, max_sen_num, max_sen_len, embedding_dim)
        :param batch:
        :return: batch (sample_text, sample_ms, sen_ms, sen_mask, labels)  must be tensor
            sample_text shape: (batch_size, max_sen_num, max_sen_len)
            sample_text_sen_mask shape: (batch_size, max_sen_num, max_sen_len)
            sample_ms shape: (batch_size, sample_metaphor)
            sen_ms shape: (batch_size, max_sen_num, sen_metaphor)
            sen_mask shape: (batch_size, max_sen_num)
            labels shape: (batch_size)
        """
        batch_padded_text = []
        batch_padded_att_mask = []
        batch_padded_sample_ms = []
        batch_padded_sen_ms = []
        batch_sen_mask = []
        batch_labels = []

        max_sen_len = -1
        max_sen_num = -1
        # bert能承载的最大长度 不超显存
        max_bert_num = 150

        for text, _, _, _ in batch:
            max_sen_num = max(max_sen_num, len(text))
            for sentence in text:
                max_sen_len = max(max_sen_len, len(sentence))
        max_sen_num = min(max_sen_num, max_bert_num)
        max_sen_len_o = min(max_sen_len, 20)
        max_sen_len = 2 + max_sen_len_o

        for sample_text, sample_m, sen_m, sample_label in batch:
            sample_text = sample_text[:max_sen_num]
            sen_m = sen_m[:max_sen_num]
            padded_text = []
            padded_att_mask = []

            for sentence in sample_text:
                sentence = sentence[:max_sen_len_o]
                temp_sen = torch.cat([torch.tensor([101]), sentence, torch.tensor([102])])
                pad_sen = torch.tensor([0] * (max_sen_len - len(temp_sen)), dtype=torch.long)
                padded_text.append(torch.cat([temp_sen, pad_sen]))
                att_mask = [1] * len(temp_sen) + [0] * (max_sen_len - len(temp_sen))
                padded_att_mask.append(torch.tensor(att_mask, dtype=torch.long))

            pad_sam_text = torch.tensor([[0]*max_sen_len]*(max_sen_num-len(sample_text)), dtype=torch.long)
            padded_sam_text = torch.cat([torch.stack(padded_text), pad_sam_text], dim=0)
            padded_sam_att_mask = torch.cat([torch.stack(padded_att_mask), pad_sam_text], dim=0)

            batch_padded_text.append(padded_sam_text)
            batch_padded_att_mask.append(padded_sam_att_mask)

            batch_padded_sample_ms.append(sample_m)

            sen_m_dim = 10
            pad_sen_m = torch.zeros((max_sen_num-len(sen_m), sen_m_dim), names=None)
            temp_sen_m = torch.cat([sen_m, pad_sen_m], dim=0)
            batch_padded_sen_ms.append(temp_sen_m)

            temp_sen_mask = [1] * len(sen_m)
            temp_sen_mask.extend([0] * (max_sen_num - len(sen_m)))
            temp_sen_mask = torch.tensor(temp_sen_mask)
            batch_sen_mask.append(temp_sen_mask)

            batch_labels.append(sample_label)

        return (torch.stack(batch_padded_text),
                torch.stack(batch_padded_att_mask),
                torch.stack(batch_padded_sample_ms),
                torch.stack(batch_padded_sen_ms),
                torch.stack(batch_sen_mask),
                torch.tensor(batch_labels, dtype=torch.long))


def print_info(matrix):

    precision = 100 * matrix[1, 1] / np.sum(matrix[1])
    recall = 100 * matrix[1, 1] / np.sum(matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (matrix[1, 1] + matrix[0, 0]) / np.sum(matrix)
    print('0: ', matrix[0])
    print('1: ', matrix[1])
    print('PRFA performance for ', precision, recall, f1, accuracy)

    result = [precision, recall, f1, accuracy]

    return np.array(result)


def evaluate(eva_dataloader, model, criterion, using_GPU=False):
    model.eval()
    count = 0
    total_loss = 0
    matrix = np.zeros((2, 2))

    for sample_text, sam_att_mask, sample_ms, sen_ms, sen_mask, labels in eva_dataloader:
        count += 1
        if using_GPU:
            sample_text = sample_text.cuda()
            sam_att_mask = sam_att_mask.cuda()
            sample_ms = sample_ms.cuda()
            sen_ms = sen_ms.cuda()
            sen_mask = sen_mask.cuda()
            labels = labels.cuda()

        predicted = model(sample_text, sam_att_mask, sample_ms, sen_ms, sen_mask, using_GPU)

        _, predict_label = torch.max(predicted, -1)
        loss = criterion(predicted, labels)
        total_loss += float(loss)

        labels = labels.data
        for i in range(len(labels)):
            p = predict_label[i]
            l = labels[i]
            matrix[p][l] += 1

    ave_loss = total_loss / count
    # print('======================= evaluation result =========================')
    # print_info(matrix)
    model.train()
    return ave_loss, print_info(matrix)


def evaluate_embed_bert(eva_dataloader, model, embed_model, criterion, using_GPU=False):
    model.eval()
    count = 0
    total_loss = 0
    matrix = np.zeros((2, 2))

    for sample_text, sam_att_mask, sample_ms, sen_ms, sen_mask, labels in eva_dataloader:
        count += 1
        if using_GPU:
            sample_text = sample_text.cuda()
            sam_att_mask = sam_att_mask.cuda()
            sample_ms = sample_ms.cuda()
            sen_ms = sen_ms.cuda()
            sen_mask = sen_mask.cuda()
            labels = labels.cuda()

        batch_size, sen_num, sen_len = sample_text.size()
        bert_i = sample_text.view(batch_size * sen_num, sen_len)
        bert_att = sam_att_mask.view(batch_size * sen_num, sen_len)
        bert_o = embed_model(input_ids=bert_i, attention_mask=bert_att)

        bert_re = bert_o.pooler_output.view(batch_size, sen_num, 768)

        predicted = model(bert_re, sample_ms, sen_ms, sen_mask, using_GPU)

        _, predict_label = torch.max(predicted, -1)
        loss = criterion(predicted, labels)
        total_loss += float(loss)

        labels = labels.data
        for i in range(len(labels)):
            p = predict_label[i]
            l = labels[i]
            matrix[p][l] += 1

    ave_loss = total_loss / count
    # print('======================= evaluation result =========================')
    # print_info(matrix)
    model.train()
    return ave_loss, print_info(matrix)