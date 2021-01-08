import torch
from transformers import BertTokenizer
import nltk


def get_metaphor_feature(data):
    """

    :param data: List[samples], sample -> tuple(List[sentences], label), sentence -> List[str]
    :return:
        sample_ms: mf of each sample (tensor) -> List[sample_m (vector)]
        vector = [sam_len, sen_num, num_meta, prop_meta, noun_meta, adj_meta, per_meta, adv_meta, vb_meta, other_meta]
        sam_sen_ms: mf of sentences of each sample -> List[sample_sen_ms], sample_sen_ms(tensor) -> List[sen_ms]
        vector = [sen_len, sen_id, num_meta, prop_meta, noun_meta, adj_meta, per_meta, adv_meta, vb_meta, other_meta]
    """
    # for testing
    # a = [0.4241,  1.0135, -1.4978, -0.0920,  0.6465,  0.6395,  0.8305,  0.5870, 0.2485,  0.8782]
    # b = [0.4241,  1.0135, -1.4978, -0.0920,  0.6465,  0.6395,  0.8305,  0.5870, 0.2485,  0.8782]
    # sample_ms = []
    # sam_sen_ms = []
    # for sample in data:
        # a = torch.randn((10, ))
        # sample_ms.append(a)
        # sen_ms = []
        # for i in sample[0]:
        #     b = torch.randn((10, ))
        #     sen_ms.append(b)
        # sam_sen_ms.append(torch.stack(sen_ms))

    # return torch.stack(sample_ms), sam_sen_ms

    """
    load metaphor identification model
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

    sample_ms = []
    sam_sen_ms = []

    for sample in data:
        sen_ms = []

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
        for sen in sample[0]:
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

            # produce metaphor feature
            for ite in range(sen_len):
                if sen_meta_re[ite] == 1:
                    sen_meta += 1
                    pos = sen_pos[ite][1]
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

            # produce sen vector
            sen_vector = torch.tensor([sen_len, sen_id, sen_meta, sen_meta/sen_len, sen_noun,
                                       sen_adj, sen_per, sen_adv, sen_vb, sen_other])
            sen_ms.append(sen_vector)

        # produce sample vector
        sam_vector = torch.tensor([total_word, num_sen, total_meta, total_meta/total_word, sam_noun,
                                   sam_adj, sam_per, sam_adv, sam_vb, sam_other])
        sample_ms.append(sam_vector)
        sam_sen_ms.append(torch.stack(sen_ms))

    return torch.stack(sample_ms), sam_sen_ms

