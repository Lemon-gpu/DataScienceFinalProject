import os
import math
import json
import gc
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad, log_softmax
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.cuda as cuda
import shutil

# 新增导入
import sacrebleu
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

UNK_ID = 0
PAD_ID = 1
SOS_ID = 2
EOS_ID = 3

batchSize = 256
numWorker = 4
numEpoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 128
d_ff = 256
n_head = 4   
encode_layer = 4
decode_layer = 4
dropout = 0.1
lr = 3e-5

maxLength = 200

trainPath: str = 'data/translation2019zh_train.json'
savePath: str = "model"

class SelfTokenizer:
    tokenizer: Tokenizer = None
    def __init__(self, language: str):
        if language == 'english':
            tokenizer_path = 'bert-base-uncased/tokenizer.json'
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        elif language == 'chinese':
            tokenizer_path = 'bert-base-chinese/tokenizer.json'
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            raise ValueError("Unsupported language.")
    def __call__(self, line: str) -> List[str]:
        return self.tokenizer.encode(line, add_special_tokens=False).tokens

class Vocabulary:
    def __init__(self, min_freq: int = 1, specials: List[str] = ['<unk>', '<pad>', '<sos>', '<eos>']):
        self.token2idx = {}
        self.idx2token = {}
        self.counter = Counter()
        self.min_freq = min_freq
        self.specials = specials

    def build_vocab(self, tokens: List[List[str]]):
        for token_list in tokens:
            self.counter.update(token_list)
        self.token2idx = {token: idx for idx, token in enumerate(self.specials)}
        self.idx2token = {idx: token for idx, token in enumerate(self.specials)}
        current_idx = len(self.specials)
        for token, freq in self.counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = current_idx
                self.idx2token[current_idx] = token
                current_idx += 1

    def __len__(self):
        return len(self.token2idx)

    def __getitem__(self, token: str) -> int:
        return self.token2idx.get(token, UNK_ID)

    def to_indices(self, tokens: List[str]) -> List[int]:
        return [self[token] for token in tokens]

    def save(self, filepath: str):
        torch.save({
            'token2idx': self.token2idx,
            'idx2token': self.idx2token
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        data = torch.load(filepath)
        vocab = cls()
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        return vocab

class TranslationDataset(Dataset):
    def __init__(self, englishSentences: torch.Tensor, chineseSentences: torch.Tensor, maxLength: int = 200):
        self.tokenizedEnglish = englishSentences
        self.tokenizedChinese = chineseSentences
        self.maxLength = maxLength
    def __len__(self):
        return len(self.tokenizedEnglish)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        englishSentence = self.tokenizedEnglish[index]
        chineseSentence = self.tokenizedChinese[index]
        return englishSentence, chineseSentence

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    englishSentences, chineseSentences = zip(*batch)
    englishSentences = torch.stack(englishSentences)
    chineseSentences = torch.stack(chineseSentences)
    src = englishSentences
    tgt = chineseSentences[:, 1:]
    tgt_y = chineseSentences[:, :-1]
    return src, tgt, tgt_y

def generatorDataloader(batchSize: int, dataset: TranslationDataset, numWorker: int = 0) -> DataLoader:
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn, num_workers=numWorker)
    return dataloader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TranslationModel(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_head: int, encode_layer: int, decode_layer: int, 
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary, dropout: float=0.1, max_length: int=200):
        super(TranslationModel, self).__init__()
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=PAD_ID)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=PAD_ID)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_length)
        self.transformer = nn.Transformer(d_model, n_head, encode_layer, decode_layer, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.predictor = nn.Linear(d_model, len(tgt_vocab))
        self.device = device
        self.tgt_vocab = tgt_vocab

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(-1)).to(self.device)
        src_key_padding_mask = self.get_key_padding_mask(src).to(self.device)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt).to(self.device)
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return out

    @staticmethod
    def get_key_padding_mask(tokens: torch.Tensor) -> torch.Tensor:
        return tokens == PAD_ID

class TranslationLoss(nn.Module):
    def __init__(self):
        super(TranslationLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum").to(device)
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = log_softmax(x, dim=-1)
        true_dist = torch.zeros_like(x).to(device)
        true_dist.scatter_(1, target.unsqueeze(1), 1)
        mask = (target == PAD_ID)
        if mask.any():
            true_dist[mask] = 0.0
        return self.criterion(x, true_dist)

def compute_metrics(preds: np.ndarray, refs: np.ndarray, idx2token: dict):
    # 转换为List[List[str]]以计算BLEU
    # 移除PAD、SOS、EOS后再进行计算
    def idxs_to_tokens(arr):
        return [idx2token[i] for i in arr if i not in [PAD_ID]]
    
    # BLEU需要预测句子和参考句子列表
    pred_sentences = [idxs_to_tokens(p) for p in preds]
    ref_sentences = [idxs_to_tokens(r) for r in refs]

    # BLEU计算
    # sacrebleu需要参考为list of list的结构
    refs_for_bleu = [[ ' '.join(r) for r in ref_sentences ]]
    hyps_for_bleu = [' '.join(p) for p in pred_sentences]
    bleu_score = sacrebleu.corpus_bleu(hyps_for_bleu, refs_for_bleu).score

    # 计算Accuracy, Precision, Recall, F1 (基于token级分类)
    # 去除PAD的评估需要对齐，这里简单粗暴地在计算时包含非PAD token
    # 实际中应对齐序列
    valid_mask = refs != PAD_ID
    valid_preds = preds[valid_mask]
    valid_refs = refs[valid_mask]

    acc = accuracy_score(valid_refs, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(valid_refs, valid_preds, average='macro', zero_division=0)

    # ROC AUC需要对logits或概率分布进行计算，这里假设已经是预测概率不可行，因为我们只保留了preds argmax。
    # 理论上，我们应该在调用此函数前提供预测的分布out_prob。
    # 为演示，这里跳过真实概率计算，直接使用二值化（不实际适用翻译任务的多类ROC/AUC）
    # 在真实应用中需在train loop中把log_softmax的结果保存下来用于计算AUC。
    # 这里仅做示例，将所有预测转为one-hot：
    # 注意：这在多类任务中有定义问题，实际中需要概率分布才能正确计算AUC。
    num_classes = max(valid_refs.max().item(), valid_preds.max().item())+1
    one_hot_refs = np.zeros((valid_refs.shape[0], num_classes))
    one_hot_preds = np.zeros((valid_preds.shape[0], num_classes))
    for i, (tr, tp) in enumerate(zip(valid_refs, valid_preds)):
        one_hot_refs[i, tr] = 1
        one_hot_preds[i, tp] = 1

    try:
        auc = roc_auc_score(one_hot_refs, one_hot_preds, average='macro', multi_class='ovr')
    except:
        # 若计算失败，则置0（可能因为某些类未出现）
        auc = 0.0

    return {
        'bleu': bleu_score,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train(epoch: int, dataloader: DataLoader, model: TranslationModel, 
          optimizer: torch.optim.Adam, criteria: TranslationLoss, batchSize: int, 
          dataSetSize: int, writer: SummaryWriter, 
          scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None):
    global_step = 0

    for i in range(epoch):
        model.train()
        all_preds = []
        all_refs = []
        for src, tgt, tgt_y in tqdm(dataloader, desc=f"Epoch {i+1}/{epoch}"):
            global_step += 1
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_y = tgt_y.type(torch.int64).to(device)
            n_tokens = (tgt_y != PAD_ID).sum()

            optimizer.zero_grad()
            out = model(src, tgt)
            out = model.predictor(out)
            loss = criteria(out.view(-1, out.size(-1)), tgt_y.view(-1)) / n_tokens
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)

            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            # 收集预测与参考
            # token-level预测
            pred_tokens = out.argmax(dim=-1).detach().cpu().numpy()
            ref_tokens = tgt_y.detach().cpu().numpy()
            all_preds.append(pred_tokens)
            all_refs.append(ref_tokens)

            if global_step % 100 == 0:
                progress = global_step * batchSize / dataSetSize * 100
                print(f'Epoch: {i+1}, Step: {global_step}, Loss: {loss.item():.4f}, Finished: {progress:.2f}%')
                with open('loss.txt', 'a+') as f:
                    f.write(f'Epoch: {i+1}, Step: {global_step}, Loss: {loss.item():.4f}, Finished: {progress:.2f}%\n')
                torch.save(model.state_dict(), 'model/translation.pt')
            gc.collect()
            cuda.empty_cache()

        # 每个epoch结束计算各项指标
        model.eval()
        all_preds = np.concatenate(all_preds, axis=0)
        all_refs = np.concatenate(all_refs, axis=0)
        metrics = compute_metrics(all_preds, all_refs, model.tgt_embedding.weight.device.type)  # 此处错误，需传idx2token
        # 我们需要 idx2token，此处从model中获取不到，需要修改main函数将tgt_vocab传入
        # 因此这里先行修改 compute_metrics 函数的调用方式并在 main 函数中传入 idx2token
        
        # 暂时留空，在 main 中补全

def main():
    # 清空log文件夹
    if os.path.exists('log'):
        shutil.rmtree('log')
    os.makedirs('log', exist_ok=True)
    
    english_vocab_path = os.path.join(savePath, 'vocab', 'english_vocab.pt')
    chinese_vocab_path = os.path.join(savePath, 'vocab', 'chinese_vocab.pt')

    if os.path.exists(english_vocab_path) and os.path.exists(chinese_vocab_path):
        englishVocab = Vocabulary.load(english_vocab_path)
        chineseVocab = Vocabulary.load(chinese_vocab_path)
    else:
        englishResult = []
        chineseResult = []
        with open(trainPath, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data, "Reading data", total=len(data)):
                line = json.loads(line)
                english: str = line['english']
                chinese: str = line['chinese']
                if chinese.isascii(): 
                    english, chinese = chinese, english
                englishResult.append(english)
                chineseResult.append(chinese)

        english_tokenizer = SelfTokenizer('english')
        chinese_tokenizer = SelfTokenizer('chinese')
        tokenized_english = [english_tokenizer(sentence) for sentence in tqdm(englishResult, total=len(englishResult), desc="Tokenizing English")]
        tokenized_chinese = [chinese_tokenizer(sentence) for sentence in tqdm(chineseResult, total=len(chineseResult), desc="Tokenizing Chinese")]

        englishVocab = Vocabulary()
        englishVocab.build_vocab(tokenized_english)
        chineseVocab = Vocabulary()
        chineseVocab.build_vocab(tokenized_chinese)

        os.makedirs(os.path.join(savePath, 'vocab'), exist_ok=True)
        englishVocab.save(english_vocab_path)
        chineseVocab.save(chinese_vocab_path)

        os.makedirs(os.path.join(savePath, 'tokenized'), exist_ok=True)
        torch.save(tokenized_english, os.path.join(savePath, 'tokenized', 'english_tokenized.pt'))
        torch.save(tokenized_chinese, os.path.join(savePath, 'tokenized', 'chinese_tokenized.pt'))

    tokenized_english_path = os.path.join(savePath, 'tokenized', 'english_tokenized.pt')
    tokenized_chinese_path = os.path.join(savePath, 'tokenized', 'chinese_tokenized.pt')

    if os.path.exists(tokenized_english_path) and os.path.exists(tokenized_chinese_path):
        tokenized_english = torch.load(tokenized_english_path)
        tokenized_chinese = torch.load(tokenized_chinese_path)
    else:
        raise FileNotFoundError("Tokenized data not found.")

    def tokens_to_padded_tensor(tokenized_sentences: List[List[str]], vocab: Vocabulary, max_length: int) -> torch.Tensor:
        indices = []
        for tokens in tokenized_sentences:
            idx = [SOS_ID] + vocab.to_indices(tokens) + [EOS_ID]
            if len(idx) > max_length:
                idx = idx[:max_length]
                idx[-1] = EOS_ID
            else:
                idx = idx + [PAD_ID] * (max_length - len(idx))
            indices.append(torch.tensor(idx, dtype=torch.int64))
        return torch.stack(indices)

    englishIndices = tokens_to_padded_tensor(tokenized_english, englishVocab, maxLength)
    chineseIndices = tokens_to_padded_tensor(tokenized_chinese, chineseVocab, maxLength)

    dataset = TranslationDataset(englishIndices, chineseIndices, maxLength)
    dataSetSize = len(dataset)
    dataloader = generatorDataloader(batchSize, dataset, numWorker)

    model = TranslationModel(d_model, d_ff, n_head, encode_layer, decode_layer, englishVocab, chineseVocab, dropout, maxLength).to(device)
    model_path = os.path.join(savePath, 'translation.pt')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criteria = TranslationLoss().to(device)
    writer = SummaryWriter(log_dir='log')

    # 重写train函数使其在epoch结束时计算指标
    def compute_metrics_ext(preds: np.ndarray, refs: np.ndarray, idx2token: dict):
        def idxs_to_tokens(arr):
            return [idx2token[i] for i in arr if i not in [PAD_ID, SOS_ID, EOS_ID]]

        pred_sentences = [idxs_to_tokens(p) for p in preds]
        ref_sentences = [idxs_to_tokens(r) for r in refs]
        refs_for_bleu = [[ ' '.join(r) for r in ref_sentences ]]
        hyps_for_bleu = [' '.join(p) for p in pred_sentences]
        bleu_score = sacrebleu.corpus_bleu(hyps_for_bleu, refs_for_bleu).score

        valid_mask = refs != PAD_ID
        valid_preds = preds[valid_mask]
        valid_refs = refs[valid_mask]

        acc = accuracy_score(valid_refs, valid_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(valid_refs, valid_preds, average='macro', zero_division=0)

        return {
            'bleu': bleu_score,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def train_modified(epoch_count, dataloader, model, optimizer, criteria, batchSize, dataSetSize, writer, scheduler):
        global_step = 0
        for i in range(epoch_count):
            model.train()
            for src, tgt, tgt_y in tqdm(dataloader, desc=f"Epoch {i+1}/{epoch_count}"):
                global_step += 1
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_y = tgt_y.type(torch.int64).to(device)
                n_tokens = (tgt_y != PAD_ID).sum()

                optimizer.zero_grad()
                out = model(src, tgt)
                out = model.predictor(out)
                loss = criteria(out.view(-1, out.size(-1)), tgt_y.view(-1)) / n_tokens
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step(loss)

                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                pred_tokens = out.argmax(dim=-1).detach().cpu().numpy()
                ref_tokens = tgt_y.detach().cpu().numpy()
                metrics = compute_metrics_ext(pred_tokens, ref_tokens, chineseVocab.idx2token)
                writer.add_scalar('eval/bleu', metrics['bleu'], global_step)
                writer.add_scalar('eval/accuracy', metrics['accuracy'], global_step)
                writer.add_scalar('eval/precision', metrics['precision'], global_step)
                writer.add_scalar('eval/recall', metrics['recall'], global_step)
                writer.add_scalar('eval/f1', metrics['f1'], global_step)
                # writer.add_scalar('eval/auc', metrics['auc'], global_step)

                if global_step % 100 == 0:
                    progress = global_step * batchSize / dataSetSize * 100
                    print(f'Epoch: {i+1}, Step: {global_step}, Loss: {loss.item():.4f}, Finished: {progress:.2f}%')
                    with open('loss.txt', 'a+') as f:
                        f.write(f'Epoch: {i+1}, Step: {global_step}, Loss: {loss.item():.4f}, Finished: {progress:.2f}%\n')
                    torch.save(model.state_dict(), 'model/translation.pt')
                gc.collect()
                cuda.empty_cache()

    with open('loss.txt', 'w') as f:
        f.truncate()

    train_modified(numEpoch, dataloader, model, optimizer, criteria, batchSize, dataSetSize, writer, scheduler)


if __name__ == '__main__':
    main()
