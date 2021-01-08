import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import time
import matplotlib.pyplot as plt

from util_bert import get_data, index_sentences, TextDataset, evaluate
from model_bert import TextClassificationModel
from early_stop_util import EarlyStopping

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
using_GPU = torch.cuda.is_available()
print(using_GPU)

"""
Data pre-process
"""
train_dp = 'train_an'
test_dp = 'test_an'

train_data, train_vocab, train_sample_ms, train_sam_sen_ms = get_data(train_dp)
test_data, test_vocab, test_sample_ms, test_sam_sen_ms = get_data(test_dp)

train_embedded_text, train_labels = index_sentences(train_data)
test_embedded_text, test_labels = index_sentences(test_data)

"""
Produce Dataset & DataLoader
"""
train_dataset = TextDataset(train_embedded_text, train_sample_ms, train_sam_sen_ms, train_labels)
test_dataset = TextDataset(test_embedded_text, test_sample_ms, test_sam_sen_ms, test_labels)

batch_size = 4
train_dataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=TextDataset.collate_fn)
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=TextDataset.collate_fn)

model = TextClassificationModel(rnn_dim=512, rnn_num_layer=2, im_dim=1024, que_dim=10, key_dim=10, deep=20, num_class=2)

if using_GPU:
    model.cuda()
model.train()

class_weights = torch.FloatTensor([1, 2]).cuda() if using_GPU else torch.FloatTensor([1, 2])  # 赋予不同权值
loss_criterion = nn.NLLLoss(weight=class_weights)
m_optimizer = optim.Adam(model.parameters())

t = time.gmtime()
f_path = str(t.tm_mon)+'_'+str(t.tm_mday)+'_'+str(t.tm_hour)
early_stopping = EarlyStopping(patience=10, path='temp_model/'+f_path+'_checkpoint.pt')

num_epochs = 10
x = []
train_loss = []
val_loss = []

for epoch in range(num_epochs):
    print()
    print("Starting epoch {}".format(epoch + 1))
    losses = []
    for sample_text, sam_att_mask, sample_ms, sen_ms, sen_mask, labels in train_dataLoader:
        if using_GPU:
            sample_text = sample_text.cuda()
            sam_att_mask = sam_att_mask.cuda()
            sample_ms = sample_ms.cuda()
            sen_ms = sen_ms.cuda()
            sen_mask = sen_mask.cuda()
            labels = labels.cuda()

        predicted = model(sample_text, sam_att_mask, sample_ms, sen_ms, sen_mask, using_GPU)
        batch_loss = loss_criterion(predicted, labels)

        model.zero_grad()
        batch_loss.backward()
        m_optimizer.step()

        losses.append(float(batch_loss))

    print('============= train loss ==============')
    ave_loss = sum(losses) / len(losses)
    print(ave_loss)

    print('============= val result ==============')
    avg_loss, result = evaluate(eva_dataloader=test_dataLoader, model=model,
                                criterion=loss_criterion, using_GPU=using_GPU)

    x.append(epoch + 1)
    train_loss.append(ave_loss)
    val_loss.append(avg_loss)

    early_stopping(ave_loss, model)
    if early_stopping.early_stop:
        print('Early stop!!!!')
        break

plt.plot(x, train_loss, color='red', linewidth=2.0, linestyle='-')
plt.plot(x, val_loss, color='blue', linewidth=2.0, linestyle='-')
plt.show()
