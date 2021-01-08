import pandas as pd
import string
import torch.nn as nn
import mmap
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import nltk
import csv
from transformers import BertTokenizer


def data_pre_process(fp_text, fp_label):
    """
    :param fp_text: file path of text
    :param fp_label: file path of label
    :return:
    List[sample]: sample -> tuple(List[sentences], label) the sentences have been processed: sentence -> List[str]
    vocab_set
    """
    pd_t = pd.read_csv(fp_text).values
    pd_l = pd.read_csv(fp_label).values
    label_dict = {}
    samples = []
    sen_len = 0
    sen_count = 0
    sam_len = 0
    sam_count = 0
    vocab = []

    for i in pd_l:
        label_dict[i[0]] = i[1]
    for i in pd_t:
        if i[0] in label_dict.keys():
            for k in range(1, 11):
                if type(i[k]) == float:
                    # print(i[0], k)
                    # print(i[k])
                    continue
                text = i[k].replace('\n', '.').replace('  ', '.').lower()
                text = text.split('.')
                # print(text)
                sentences = []
                for t in text:
                    # temp = nltk.word_tokenize(t)
                    temp = t.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation})).split()
                    if len(temp) > 5:
                        sen_len += len(temp)
                        sen_count += 1
                        sentences.append(temp)
                        vocab.extend(temp)
                        # print(temp)
                # print(sentences)
                if len(sentences) > 2:
                    sam_len += len(sentences)
                    sam_count += 1
                    samples.append((sentences, label_dict[i[0]]))
    print("total token: ", sen_len, "total sentences: ", sam_len)
    print("ave token in sentence: ", sen_len/sen_count, "ave sentences in sample: ", sam_len/sam_count)
    return samples, set(vocab)


def create_data(fp_text, fp_label):
    """
    from data to metaphor_re. Need to be run before experiment
    :param fp_text: get text data
    :param fp_label: get label data
    :return:
        produce metaphor_re
    """
    pd_t = pd.read_csv(fp_text).values
    pd_l = pd.read_csv(fp_label).values
    label_dict = {}
    sam_id = -1

    """
    metaphor part
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    m_model = torch.load('model/final_19.pt', map_location=torch.device('cpu'))
    m_model.eval()

    """
    POS tags
    """
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    adj_tags = ['JJ', 'JJS', 'JJR']
    per_tags = ['POS', 'PRP', 'PRP$']
    adv_tags = ['RB', 'RBR', 'RBS']
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    """
    file writer
    样本号，标签，样本隐喻特征
    """
    f_sam = open('metaphor_re/sam_label_meta.csv', 'w', encoding='utf-8', newline='')
    w_sam = csv.writer(f_sam)
    # 隐喻词语记录
    f1 = open('metaphor_re/positive.txt', 'w')
    f2 = open('metaphor_re/negative.txt', 'w')

    for i in pd_l:
        label_dict[i[0]] = i[1]
    for j in pd_t:
        if j[0] in label_dict.keys():
            for k in range(1, 11):
                if type(j[k]) == float:
                    # print(i[0], k)
                    # print(i[k])
                    continue

                text = j[k].replace('\n', '.').replace('  ', '.').lower()
                # text = j[k].replace('  ', '\n').lower()
                text = text.split('.')
                # print(text)
                if len(text) > 2:
                    sam_id += 1

                    # 样本句子
                    f_sam_sen = open('metaphor_re/sens/'+str(sam_id)+'_sen.csv', 'w', encoding='utf-8', newline='')
                    w_sam_sen = csv.writer(f_sam_sen)
                    # 句子隐喻特征
                    f_sam_meta = open('metaphor_re/metaphor/'+str(sam_id)+'_meta.csv', 'w', encoding='utf-8', newline='')
                    w_sam_meta = csv.writer(f_sam_meta)

                    num_sen = 0
                    total_word = 0
                    total_meta = 0

                    sam_noun = 0
                    sam_adj = 0
                    sam_per = 0
                    sam_adv = 0
                    sam_vb = 0
                    sam_other = 0

                    sen_id = -1
                    for t in text:
                        sen = nltk.word_tokenize(t)
                        if 500 > len(sen) > 10:
                            w_sam_sen.writerow(sen)
                            sen_id += 1
                            sen_len = len(sen)
                            sen_meta = 0

                            sen_noun = 0
                            sen_adj = 0
                            sen_per = 0
                            sen_adv = 0
                            sen_vb = 0
                            sen_other = 0

                            sen_pos = nltk.pos_tag(sen)

                            # metaphor identification part
                            meta_ids = tokenizer.convert_tokens_to_ids(sen)
                            i = [101]
                            i.extend(meta_ids)
                            i.append(102)

                            l = len(i)
                            i = torch.tensor([i])
                            tt = torch.tensor([[0]*l])
                            am = torch.tensor([[1]*l])
                            po = torch.tensor([[k for k in range(l)]])
                            o = m_model(input_ids=i, token_type_ids=tt, attention_mask=am, position_ids=po)
                            _, label = torch.max(o.logits, dim=-1)
                            sen_meta_re = label[0][1:-1]

                            # print(sen_meta_re)

                            # produce metaphor feature
                            for ite in range(sen_len):
                                if sen_meta_re[ite] == 1:
                                    sen_meta += 1
                                    pos = sen_pos[ite][1]
                                    word = sen_pos[ite][0]

                                    if label_dict[j[0]]:
                                        f1.write(word+' '+pos+'\n')
                                    else:
                                        f2.write(word+' '+pos+'\n')

                                    if pos in noun_tags:
                                        sen_noun += 1
                                    elif pos in adj_tags:
                                        sen_adj += 1
                                    elif pos in per_tags:
                                        sen_per += 1
                                    elif pos in adv_tags:
                                        sen_adv += 1
                                    elif pos in vb_tags:
                                        sen_vb += 1
                                    else:
                                        sen_other += 1

                            # update sam feature
                            num_sen += 1
                            total_word += sen_len
                            total_meta += sen_meta

                            sam_noun += sen_noun
                            sam_adj += sen_adj
                            sam_per += sen_per
                            sam_adv += sen_adv
                            sam_vb += sen_vb
                            sam_other += sen_other

                            sen_vector = [sen_len, sen_id, sen_meta, sen_meta/sen_len, sen_noun,
                                          sen_adj, sen_per, sen_adv, sen_vb, sen_other]

                            w_sam_meta.writerow(sen_vector)
                            print(sam_id, sen_id)
                    if total_word > 0:
                        sam_vector = [sam_id, label_dict[j[0]], total_word, num_sen, total_meta, total_meta/total_word, sam_noun,
                                      sam_adj, sam_per, sam_adv, sam_vb, sam_other]
                        print(sam_vector)
                        w_sam.writerow(sam_vector)


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


def get_word2idx_idx2word(vocab):
    """

    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_embedding_matrix(word2idx, idx2word, normalization=False):
    """
    assume padding index is 0

    :param word2idx: a dictionary: string --> int, includes <PAD> and <UNK>
    :param idx2word: a dictionary: int --> string, includes <PAD> and <UNK>
    :param normalization:
    :return: an embedding matrix: a nn.Embeddings
    """
    # Load the GloVe vectors into a dictionary, keeping only words in vocab
    embedding_dim = 300
    # glove_path = "glove/glove.840B.300d.txt"
    glove_path = "glove/testvec.txt"
    glove_vectors = {}
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(glove_path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector

    print("Number of pre-trained word vectors loaded: ", len(glove_vectors))

    # Calculate mean and stdev of embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)

    # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
    # with a similar distribution as the pretrained embeddings for words in vocab.
    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix, requires_grad=False)
    return embeddings


def embed_sentences(data, word2idx, glove_embedding):
    """
    对已分词数据进行嵌入
    :param data: List[samples], sample -> tuple(List[sentences], label)
    :param word2idx: dict[word->int]
    :param glove_embedding: nn.Embedding
    :return:
        embedded data, List[embedded_text] and List[label], embedded_text -> List[sentences], sentence -> List[vectors]
    """
    index_data = []
    labels = []
    for sample in data:
        index_sample = []
        for sentence in sample[0]:
            temp = torch.tensor([word2idx.get(x, 1) for x in sentence])
            temp_embed = glove_embedding(temp)
            index_sample.append(temp_embed)
        index_data.append(index_sample)
        labels.append(sample[1])
    # index_data = torch.tensor(index_data, dtype=torch.long, requires_grad=False)
    #
    # embedded_text = glove_embedding(index_data)
    embedded_text = index_data
    labels = torch.tensor(labels)
    return embedded_text, labels


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
        填充数据 按最大数据尺寸处理 sample_text.shape = (batch_size, max_sen_num, max_sen_len, embedding_dim)
        :param batch:
        :return: batch (sample_text, sample_ms, sen_ms, sen_mask, labels)  must be tensor
            sample_text shape: (batch_size, max_sen_num, max_sen_len, embedding_dim)
            sample_ms shape: (batch_size, sample_metaphor)
            sen_ms shape: (batch_size, max_sen_num, sen_metaphor)
            sen_mask shape: (batch_size, max_sen_num)
            labels shape: (batch_size)
        """
        batch_padded_text = []
        batch_padded_sample_ms = []
        batch_padded_sen_ms = []
        batch_sen_mask = []
        batch_labels = []

        max_sen_num = -1
        max_sen_len = -1

        for text, _, _, _ in batch:
            max_sen_num = max(max_sen_num, len(text))
            for sentence in text:
                max_sen_len = max(max_sen_len, len(sentence))

        for sample_text, sample_m, sen_m, sample_label in batch:

            padded_text = []
            embedding_dim = 300

            for sentence in sample_text:
                pad_sentence = torch.zeros((max_sen_len-len(sentence), embedding_dim), names=None)
                temp_sen = torch.cat([sentence, pad_sentence], dim=0)
                padded_text.append(temp_sen)

            pad_sample_text = torch.zeros((max_sen_num-len(sample_text), max_sen_len, embedding_dim), names=None)
            # 要的是这个
            temp_sample_text = torch.cat([torch.stack(padded_text), pad_sample_text], dim=0)
            batch_padded_text.append(temp_sample_text)

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
                torch.stack(batch_padded_sample_ms),
                torch.stack(batch_padded_sen_ms),
                torch.stack(batch_sen_mask),
                torch.tensor(batch_labels, dtype=torch.long))


def print_info(matrix):
    """
    Prints the precision, recall, f1, and accuracy for each pos tag
    Assume that the confusion matrix is implicitly mapped with the idx2pos
    i.e. row 0 in confusion matrix is for the pos tag mapped by int 0 in idx2pos

    :param matrix: a confusion matrix of shape (#pos_tags, 2, 2)
    :param idx2pos: idx2pos: a dictionary: int --> pos tag
    :return: a matrix (#allpostags, 4) each row is the PRFA performance for a pos tag
    """

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

    for sample_text, sample_ms, sen_ms, sen_mask, labels in eva_dataloader:
        count += 1
        if using_GPU:
            sample_text = sample_text.cuda()
            sample_ms = sample_ms.cuda()
            sen_ms = sen_ms.cuda()
            sen_mask = sen_mask.cuda()
            labels = labels.cuda()

        predicted = model(sample_text, sample_ms, sen_ms, sen_mask, using_GPU)

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
