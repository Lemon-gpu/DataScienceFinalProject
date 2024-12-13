import pandas as pd
import jieba 
import os
from pandarallel import pandarallel

class vocabulary:

    positive_vocab: list = None
    negative_vocab: list = None
    file: str = None

    def __init__(self, file: str = 'vocabulary/ChineseFinancialSentimentDictionary.xlsx') -> None:
        '''
        Args:
            file (str): path to the file

        Returns:
            None

        '''
        self.file = file
        self.negative_vocab, self.positive_vocab = self.get_vocabulary()

    def get_vocabulary(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        By entering the path to the file, the function returns a tuple of DataFrames, 
        which include negative and positive words in order.

        Args:
            file (str): path to the file

        Returns:
            tuple[pd.DataFrame]: tuple of DataFrames, which include negative and positive words in order

        '''

        if self.negative_vocab is not None and self.positive_vocab is not None:
            return self.negative_vocab, self.positive_vocab
        
        def string_handle(x: str) -> str:
            x = x.str
            x = x.replace('\n', '')
            x = x.replace(' ', '')
            x = x.replace('\t', '')
            return x
        
        negative: pd.DataFrame = pd.read_excel(self.file, sheet_name='negative')
        positive: pd.DataFrame = pd.read_excel(self.file, sheet_name='positive')
        negative = negative.apply(string_handle, axis=0)
        
        positive = positive.apply(string_handle, axis=0)
        return negative.iloc[:, 0].tolist(), positive.iloc[:, 0].tolist()


class count_score:

    vocab: vocabulary = None

    def __init__(self, vocab: vocabulary = None) -> None:
        if vocab is None:
            self.vocab = vocabulary()
        else:
            self.vocab = vocab

    def count_positive_score(self, content: str) -> float:
        '''
        Count the number of positive words in the content

        Args:
            content (str): content to count
            vocab (vocabulary): vocabulary object

        Returns:
            float: number of positive words

        '''
        
        positive = self.vocab.positive_vocab
        content: list = jieba.lcut(content)
        positive = [i for i in positive if i in content]
        return len(positive) / len(content)

    def count_negative_score(self, content: str) -> float:
        '''
        Count the number of negative words in the content

        Args:
            content (str): content to count
            vocab (vocabulary): vocabulary object

        Returns:
            float: number of negative words

        '''
        negative = self.vocab.negative_vocab
        content: list = jieba.lcut(content)
        negative = [i for i in negative if i in content]
        return len(negative) / len(content)

    def count_score(self, content: str) -> float:
        '''
        Count the score of the content. 
        When positive > negative, the content is positive, max = 1
        When positive < negative, the content is negative, min = -1
        When positive == negative, the content is neutral, equal to 0

        Args:
            content (str): content to count
            vocab (vocabulary): vocabulary object

        Returns:
            float: score of the content

        '''
        positive = self.count_positive_score(content)
        negative = self.count_negative_score(content)
        return positive - negative # when positive > negative, the content is positive, max = 1
        # when positive < negative, the content is negative, min = -1
        # when positive == negative, the content is neutral, equal to 0

    def count_all_score(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Count the score of all content in the DataFrame

        Args:
            data (pd.DataFrame): DataFrame to count

        Returns:
            pd.DataFrame: DataFrame with the score of all content

        '''
            
        data['score'] = data.parallel_apply(lambda x: self.count_score(str(x[0]) + ' ' + str(x[1])), axis=1)
        data.sort_values(by='score', inplace=True, ascending=False) # from high to low
        for i in range(len(data)):
            if i >= 0 and i <= len(data) / 3:
                data.iloc[i, 2] = 2
            elif i >= len(data) / 3 and i <= len(data) * 2 / 3:
                data.iloc[i, 2] = 1
            else:
                data.iloc[i, 2] = 0
        data['content'] = data['title'] + ' ' + data['content']
        data.drop(['title'], axis=1, inplace=True)
        return data

def read_datas(files_path: str) -> pd.DataFrame:
    '''
    Read data from the file

    Args:
        file (str): path to the file

    Returns:
        pd.DataFrame: DataFrame of the data

    '''
    data: pd.DataFrame = pd.DataFrame()
    # open all file in the folder
    for file in os.listdir(files_path):
        # read data from fil
        path = os.path.join(files_path, file)
        data = pd.read_excel(path)
        data = pd.concat((data, path), ignore_index=True)
    return data

def main():
    pandarallel.initialize()
    # read data
    data = read_datas('dataset/Chinese')
    print(data.dtypes)
    # count score
    count = count_score()
    data = count.count_all_score(data)
    # save data
    data.to_excel('dataset/ChineseResult.xlsx', index=False, encoding='utf8')

if __name__ == '__main__':
    main()