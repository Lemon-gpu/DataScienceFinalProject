import csv
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer
from torch import nn
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import transformers
import gc
import shutil
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

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
    result = result.sample(frac=1).reset_index(drop=True) 
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
    def __init__(self, output_dim: int, pretrained_name: str='bert-base-chinese'):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name).train()
        self.mlp = nn.Linear(768, output_dim)
    def forward(self, tokens_X: transformers.tokenization_utils_base.BatchEncoding):
        temp = self.bert(**tokens_X)
        return self.mlp(temp[1])


class SentimentDataset(Dataset):
    def __init__(self, comments: list[str], labels: list[int]):
        self.comments = comments
        self.labels = labels

    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx) -> tuple[str, int]:
        comment = self.comments[idx]
        label = self.labels[idx]
        if comment != comment: # 判断是否为nan
            comment = ''
        return comment, label
    
def get_Dataloader(comments: list[str], labels: list[int], tokenizer: BertTokenizer, batch: int):

    def collate_fn(batch):
        comments, labels = zip(*batch)
        tokens_X = tokenizer(list(comments), padding='max_length', truncation=True, max_length=100, return_tensors='pt')
        return tokens_X, torch.tensor(labels)

    dataset = SentimentDataset(comments, labels)
    return DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=3)

def train_bert_classifier(net: BERTClassifier, loss_fn: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, 
                          scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                          train_dataloader: DataLoader, device: torch.device, epochs: int, data_size: int, writer: SummaryWriter):
    global_step = 0
    net.train()
    for epoch in range(epochs):
        progress_bar = tqdm(desc=f'training epoch {epoch+1}/{epochs}', total=data_size)
        for tokens_X, y in train_dataloader:
            tokens_X = tokens_X.to(device=device)
            y = y.to(device=device)
            
            optimizer.zero_grad()
            res = net(tokens_X)
            l = loss_fn(res, y)
            l.backward()
            optimizer.step()
            scheduler.step(l)

            # 计算该batch的各项指标
            preds = res.argmax(dim=1).cpu().tolist()
            labels = y.cpu().tolist()
            acc = sum([1 if p == t else 0 for p,t in zip(preds, labels)]) / len(labels)
            precision = precision_score(labels, preds, average='macro', zero_division=0)
            recall = recall_score(labels, preds, average='macro', zero_division=0)
            f1 = f1_score(labels, preds, average='macro', zero_division=0)

            lr = optimizer.param_groups[0]['lr']

            # 在每个batch结束后记录tensorboard
            writer.add_scalar('Train/Loss', l.item(), global_step)
            writer.add_scalar('Train/LR', lr, global_step)
            writer.add_scalar('Train/Accuracy', acc, global_step)
            writer.add_scalar('Train/Precision', precision, global_step)
            writer.add_scalar('Train/Recall', recall, global_step)
            writer.add_scalar('Train/F1', f1, global_step)

            progress_bar.set_postfix({'loss': l.item(), 'lr': lr, 'acc': acc, 'f1': f1})
            progress_bar.update(len(y))
            global_step += 1

            torch.cuda.empty_cache()
            gc.collect()

        # 不在每个epoch结束后做记录，所有记录都在batch结束后完成
        # 保存模型（可选）
        torch.save(net, 'model/model.pth')


def main(path: str, split: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 350
    epochs = 3

    # 清空log文件夹
    if os.path.exists('logs'):
        shutil.rmtree('logs')
    os.makedirs('logs', exist_ok=True)

    writer = SummaryWriter('logs')

    train_comments, train_labels, test_comments, test_labels = read_files(path, split)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese') 

    train_dataset = get_Dataloader(train_comments, train_labels, tokenizer, batch_size)
    # 测试集可以不用loader，如果不需要在epoch结束后记录指标则这里不必须
    # test_dataset = get_Dataloader(test_comments, test_labels, tokenizer, batch_size)

    '''
    if os.path.exists('model/model.pth'):
        net = torch.load('model/model.pth')
        print('Model loaded.')
    else: 
        net = BERTClassifier(output_dim=3).to(device)  
    '''
    net = BERTClassifier(output_dim=3).to(device)  
                       
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    # 开始训练，并使用tensorboard记录（每个batch）
    train_bert_classifier(net, loss, optimizer, scheduler, train_dataset, device, epochs, len(train_comments), writer)
    writer.close()

if __name__ == '__main__':
    main('dataset/train', 0.9)
