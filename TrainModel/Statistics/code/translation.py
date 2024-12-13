import torch
import torch.nn as nn
import os
from pathlib import Path
from torchtext.vocab import Vocab
import pandas as pd
from tokenizers import Tokenizer
import math
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class TranslationModel(nn.Module):

    def __init__(self, d_model, src_vocab, tgt_vocab, dropout=0.1):
        super(TranslationModel, self).__init__()

        # 定义原句子的embedding
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        # 定义目标句子的embedding
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        # 定义posintional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=72) # 写死了最大长度为72
        # 定义Transformer
        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)

        # 定义最后的预测层，这里并没有定义Softmax，而是把他放在了模型外。
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        """
        进行前向传递，输出为Decoder的输出。注意，这里并没有使用self.predictor进行预测，
        因为训练和推理行为不太一样，所以放在了模型外面。
        :param src: 原batch后的句子，例如[[0, 12, 34, .., 1, 2, 2, ...], ...]
        :param tgt: 目标batch后的句子，例如[[0, 74, 56, .., 1, 2, 2, ...], ...]
        :return: Transformer的输出，或者说是TransformerDecoder的输出。
        """

        """
        生成tgt_mask，即阶梯型的mask，例如：
        [[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]]
        tgt.size()[-1]为目标句子的长度。
        """
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # 掩盖住原句子中<pad>的部分，例如[[False,False,False,..., True,True,...], ...]
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        # 掩盖住目标句子中<pad>的部分
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        return tokens == 2


class Translation:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module  = torch.load('model/model.pt', map_location=device)
    enVocab: Vocab = torch.load('model/vocab_en.pt', map_location=device)
    zhVocab: Vocab = torch.load('model/vocab_zh.pt', map_location=device)
    tokenizer: Tokenizer = Tokenizer.from_pretrained('bert-base-uncased')
    maxLength = 72

    def cut(self, src: str) -> list[torch.Tensor]:
        result = []
        for i in range(0, len(src), self.maxLength - 2):
            temp = src[i: min(i + self.maxLength - 2, len(src))]
            temp = torch.tensor([0] + self.enVocab(self.enTokenizer(temp)) + [1]).unsqueeze(0).to(self.device)
            result.append(temp)
        return result
    
    def translation(self, src: str) -> str:
        # 将与原句子分词后，通过词典转为index，然后增加<bos>和<eos>
        src = self.cut(src)
        result = ''
        for s in src:
            tgt = torch.tensor([[0]]).to(self.device)
            # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
            for i in range(self.maxLength):
                # 进行transformer计算
                out = self.model(s, tgt)
                # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
                predict = self.model.predictor(out[:, -1])
                # 找出最大值的index
                y = torch.argmax(predict, dim=1)
                # 和之前的预测结果拼接到一起
                tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
                # 如果为<eos>，说明预测结束，跳出循环
                if y == 1:
                    break
            # 将预测tokens拼起来
            tgt = ''.join(self.zhVocab.lookup_tokens(tgt.squeeze().tolist())).replace("<s>", "").replace("</s>", "")
            result += tgt
        return result
    
    def enTokenizer(self, text: str) -> list[str]:
        return self.tokenizer.encode(text, add_special_tokens=False).tokens
    
    def __call__(self, text: str) -> str:
        return self.translation(text)


def sentimentClassification(text: str) -> int:
    if text.__contains__('positive'): # 我知道可以用in，但我不习惯
        return 2
    elif text.__contains__('negative'):
        return 1
    else:
        return 0

def getFileLens(filePath: str) -> int:
    result: int = 0
    with open(filePath, "r", encoding="utf8") as f: # 得用ansi编码，不然会报错
        for line in f:
            result += 1
    return result

def readFile(filePath: str) -> list[dict]:
    fileLens: int = getFileLens(filePath)
    result: list[dict] = []
    translator: Translation = Translation()
    with open(filePath, "r", encoding="utf8") as f: # 得用ansi编码，不然会报错
        for line in tqdm(f, desc=filePath, total=fileLens):
            segment: list[str] = line.split("@")
            content: str = ''.join(segment[0: -1])
            content = translator.translation(content)
            sentiment = sentimentClassification(segment[-1])
            result.append({
                "content": content,
                "sentiment": sentiment
            })
    return result

def readFiles(filePath: str, savePath: str):
    for file in os.listdir(filePath):
        if file.split('.')[-1] != 'txt':
            continue
        temp = readFile(os.path.join(filePath, file))
        temp  = pd.DataFrame(temp)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        temp.to_csv(os.path.join(savePath, file.replace('.txt', '.csv')), index=False)



if __name__ == "__main__":
    readFiles('dataset/validation', 'dataset/Englsh/data/result')
    print("Done!")

