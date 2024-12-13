import csv
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import transformers
import gc
"""
读取评论文件的评论信息
"""
def read_files(filePath: str, split: float, cache: bool = True) \
        -> tuple[list[str], list[int], list[str], list[int]]:
    if cache:
        if os.path.exists(os.path.join(filePath, 'result.csv')):
            result = pd.read_csv(os.path.join(filePath, 'result.csv'))
            train_comments = result['content'].tolist()[:int(len(result) * split)]
            train_labels = result['sentiment'].tolist()[:int(len(result) * split)]
            test_comments = result['content'].tolist()[int(len(result) * split):]
            test_labels = result['sentiment'].tolist()[int(len(result) * split):]
            print('Cached data loaded.')
            return train_comments, train_labels, test_comments, test_labels
    
    result = pd.DataFrame()
    for file in os.listdir(filePath):
        if file.endswith(".csv"):
            temp = pd.read_csv(os.path.join(filePath, file))
            result = pd.concat([result, temp])

    # 打乱数据
    result = result.sample(frac=1).reset_index(drop=True) # frac=1表示打乱数据，reset_index重置索引
    if cache:
        result.to_csv(os.path.join(filePath, 'result.csv'), index=False)

    # 划分训练集与测试集
    train_comments = result['content'].tolist()[:int(len(result) * split)]
    train_labels = result['sentiment'].tolist()[:int(len(result) * split)]
    test_comments = result['content'].tolist()[int(len(result) * split):]
    test_labels = result['sentiment'].tolist()[int(len(result) * split):]
    return train_comments, train_labels, test_comments, test_labels


"""
Step2: 定义BERTClassifier分类器模型
"""

class BERTClassifier(nn.Module):
    # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
    def __init__(self, output_dim: int, pretrained_name: str='bert-base-chinese'):
        super(BERTClassifier, self).__init__()
        # 定义 Bert 模型
        self.bert = BertModel.from_pretrained(pretrained_name).train()
        # 外接全连接层
        self.mlp = nn.Linear(768, output_dim)
    def forward(self, tokens_X: transformers.tokenization_utils_base.BatchEncoding):
        # tokens_X 为字典类型，包含input_ids, token_type_ids, attention_mask三个key
        # self.temp(**tokens_X)
        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        temp = self.bert(**tokens_X)
        # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析 
        return self.mlp(temp[1])


class SentimentDataset(Dataset):
    def __init__(self, comments: list[str], labels: list[int]):
        self.comments = comments
        self.labels = labels

    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        comment = self.comments[idx]
        label = self.labels[idx]
        if comment != comment: # 判断是否为nan
            comment = ''
        return comment, label
    
def get_Dataloader(comments: list[str], labels: list[int], tokenizer: BertTokenizer, batch: int):

    def collate_fn(batch):
        comments, labels = zip(*batch)
        tokens_X =  tokenizer(comments, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
        return tokens_X, torch.tensor(labels)

    dataset = SentimentDataset(comments, labels)
    return DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=3)

"""
评估函数，用以评估数据集在神经网络下的精确度
"""

def evaluate(net: BERTClassifier, dataloader: DataLoader, device: torch.device):
    net.eval()
    sum_correct = 0
    count = 0

    progress_bar = tqdm(desc='evaluating', total=len(dataloader.dataset))
    for tokens_X, y in dataloader:
        tokens_X = tokens_X.to(device=device)
        y = y.to(device=device)
        res = net(tokens_X)
        sum_correct += (res.argmax(axis=1) == y).sum().item()
        count += len(y)
        progress_bar.update(len(y))
        progress_bar.set_postfix({'current_acc': sum_correct/count})
        torch.cuda.empty_cache()
        gc.collect()
    
    return sum_correct/len(dataloader.dataset)

def train_bert_classifier(net: BERTClassifier, loss: nn.CrossEntropyLoss, optimizer: torch.optim.SGD, 
                          scheduler: torch.optim.lr_scheduler,
                          dataloader: DataLoader, device: torch.device, epochs: int, data_size: int):
    net.train()
    progress_bar = tqdm(desc='training', total=data_size * epochs)
    for i in range(epochs):
        count = 0
        for tokens_X, y in dataloader:
            tokens_X = tokens_X.to(device=device)
            y = y.to(device=device)
            res = net(tokens_X)
            l = loss(res, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step(l)

            progress_bar.set_postfix({'loss': l.item(), 'lr': optimizer.param_groups[0]['lr']})
            progress_bar.update(len(y))

            if count % 1000 == 0:
                torch.save(net, 'model/model.pth')

            count += len(y)

            torch.cuda.empty_cache()
            gc.collect()

def test(model: BERTClassifier, device: torch.device, text: str) -> str:
    '''
    2 代表积极
    1 代表中性
    0 代表消极
    '''
    results: dict = {
        2: "积极",
        1: "中性",
        0: "消极"
    }
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') 
    model.eval()
    text = text[0: 510] if len(text) > 510 else text
    input = tokenizer(text, return_tensors="pt", padding=True).to(device)
    output = model(input)
    return results[output.argmax(dim=1).item()]

def main(path: str, split: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    # epochs = 200

    # train_comments, train_labels, test_comments, test_labels = read_files(path, split)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') 

    # train_dataset = get_Dataloader(train_comments, train_labels, tokenizer, batch_size)
    # test_dataset = get_Dataloader(test_comments, test_labels, tokenizer, batch_size)

    net = None
    if os.path.exists('eval/model.pth'):
        net = torch.load('eval/model.pth')
        print('Model loaded.')
    else: 
        net = BERTClassifier(output_dim=3).to(device)   
                       
    text: str = r'''
北京时间11月7日（周四）20:00，英国央行将公布利率决议。

在9月份的会议上，英国央行将利率保持在5.0%不变，并表示将对未来的降息保持谨慎。尽管如此，英国央行行长贝利近日表示，如果数据继续表明通胀有所进展，他们可能需要更积极地降息。事实上，英国9月整体CPI从2.2%降至1.7%，而核心CPI从3.6%降至3.2%。

利率市场定价显示，英国央行本次降息25个基点的可能性高达80%，但该行在12月再次降息25个百分点的可能性只有30%。

在英国央行本次议息会议上，市场焦点可能会落在表决结果和政策制定者的沟通上。如果投票结果显示是一个势均力敌的决定，并且会议声明再次指出不急于进一步降息，可能打压未来的宽松预期。
'''
    # loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    # scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    # train_bert_classifier(net, loss, optimizer, scheduler, train_dataset, device, epochs, len(train_comments))
    # print(evaluate(net, test_dataset, device))
    print(test(net, device, text))

if __name__ == '__main__':
    main('dataset/train', 0.9)
