import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import jieba
import re
import os
from matplotlib import font_manager
import shutil

# 设置中文字体
font_path = '/root/autodl-fs/Code/TrainModel/BERT/simhei.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# Create 'visualization_data' folder if it doesn't exist
if os.path.exists('visualization_data'):
    shutil.rmtree('visualization_data')
os.makedirs('visualization_data')
    
# ==============================
# Step 1: Load and Preprocess Data
# ==============================
df = pd.read_csv('dataset/train/result.csv', encoding='utf-8')

# Map sentiment values to labels
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df['sentiment_label'] = df['sentiment'].map(sentiment_map)

# Drop rows with empty or NaN content
df = df.dropna(subset=['content'])
df = df[df['content'].str.strip() != '']

# Basic cleanup for text (remove punctuation, line breaks, etc.)
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
    return text.strip()

df['cleaned_content'] = df['content'].apply(clean_text)

# Tokenization using jieba (for Chinese text)
df['tokens'] = df['cleaned_content'].apply(lambda x: list(jieba.cut(x)))

# You may define a list of stopwords (for demonstration, using a minimal set)
stopwords = set(['的', '和', '在', '于', '是', '了', '有', '其', '等', '年'])

# Filter out stopwords and short tokens
df['tokens'] = df['tokens'].apply(lambda words: [w for w in words if w not in stopwords and len(w) > 1])

# 构建每个情感对应的词语集合
sentiment_words = {}
for sentiment in ['Negative', 'Neutral', 'Positive']:
    subset = df[df['sentiment_label'] == sentiment]
    words = set(word for tokens in subset['tokens'] for word in tokens)
    sentiment_words[sentiment] = words

# 1. 出现在所有三个情感中的词语
common_all = sentiment_words['Negative'] & sentiment_words['Neutral'] & sentiment_words['Positive']

# 2. 出现在正面和负面情感中的词语
common_neg_pos = (sentiment_words['Negative'] & sentiment_words['Positive']) - common_all

# 3. 出现在负面和中性情感中的词语
common_neg_neu = (sentiment_words['Negative'] & sentiment_words['Neutral']) - common_all

# 4. 出现在中性和正面情感中的词语
common_neu_pos = (sentiment_words['Neutral'] & sentiment_words['Positive']) - common_all

# 合并需要移除的词语
words_to_remove = common_all.union(common_neg_pos, common_neg_neu, common_neu_pos)

# 从 tokens 中移除这些词语
df['tokens'] = df['tokens'].apply(lambda words: [w for w in words if w not in words_to_remove])

# 从 sentiment_words 中移除这些词语
for sentiment in sentiment_words:
    sentiment_words[sentiment] = sentiment_words[sentiment] - words_to_remove
    # 修改过滤条件，使用正确的逻辑去除包含字母或数字的词语
    for word in sentiment_words[sentiment].copy():
        for c in word.lower():
            if c >= 'a' and c <= 'z' or c >= '0' and c <= '9':
                sentiment_words[sentiment].remove(word)
                break

# ==============================
# Step 2: Sentiment Distribution
# ==============================
sentiment_counts = df['sentiment_label'].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['red', 'gray', 'green'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Entries")
plt.xticks(rotation=0)
for i, count in enumerate(sentiment_counts):
    plt.text(i, count + 0.5, str(count), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('visualization_data/sentiment_distribution.png', dpi=500)
plt.close()

# ==============================
# Step 3: Text Length Distribution by Sentiment
# ==============================
# Calculate text lengths
df['length'] = df['cleaned_content'].apply(len)

# Boxplot of text length by sentiment
plt.figure(figsize=(8,6))
df.boxplot(column='length', by='sentiment_label', grid=False)
plt.title("Text Length Distribution by Sentiment")
plt.suptitle("")  # Remove default title
plt.xlabel("Sentiment")
plt.ylabel("Text Length (characters)")
plt.savefig('visualization_data/text_length_boxplot.png', dpi=500)
plt.close()

# Alternatively, a histogram could be used, layered by sentiment
plt.figure(figsize=(10,6))
for sentiment in ['Negative', 'Neutral', 'Positive']:
    subset = df[df['sentiment_label'] == sentiment]
    plt.hist(subset['length'], bins=20, alpha=0.5, label=sentiment)
plt.title("Text Length Distribution Histogram by Sentiment")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig('visualization_data/text_length_histogram.png', dpi=500)
plt.close()

# ==============================
# Step 4: Most Common Words by Sentiment
# ==============================
def top_words_for_sentiment(df, sentiment, top_n=10):
    # Filter for the given sentiment
    subset = df[df['sentiment_label'] == sentiment]
    # Flatten token lists
    all_words = [word for tokens in subset['tokens'] for word in tokens]
    counter = Counter(all_words)
    return counter.most_common(top_n)

for sentiment in ['Negative', 'Neutral', 'Positive']:
    common_words = top_words_for_sentiment(df, sentiment)
    words, counts = zip(*common_words)
    plt.figure(figsize=(8,6))
    # 设置颜色图谱为rainbow
    plt.barh(words, counts, color=plt.cm.rainbow(range(10)))
    plt.title(f"Top Words in {sentiment} Content")
    plt.xlabel("Frequency")
    plt.gca().invert_yaxis()  # highest at top
    plt.tight_layout()
    filename = f'visualization_data/top_words_{sentiment.lower()}.png'
    plt.savefig(filename, dpi=500)
    plt.close()

# ==============================
# Step 5: Word Clouds
# ==============================
# Generate a single word cloud function
# 在词云函数中确保指定字体路径
def generate_wordcloud(text, title=None, filename=None):
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        font_path='/root/autodl-fs/Code/TrainModel/BERT/simhei.ttf',  # 指定中文字体路径
        colormap='rainbow'
    ).generate(' '.join(text))
    plt.figure(figsize=(16,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    if title:
        plt.title(title, fontsize=24)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()

for sentiment in ['Negative', 'Neutral', 'Positive']:
    sentiment_word = sentiment_words[sentiment]
    generate_wordcloud(sentiment_word, title=f"{sentiment} Word Cloud", filename=f'visualization_data/wordcloud_{sentiment.lower()}.png')
# ==============================
# Additional Ideas:
# ==============================
# - You can create a scatter plot or dimensionality reduction visualization (e.g., using TF-IDF vectors and TSNE/UMAP) 
#   to see if there's any clustering of sentiments.
# - You can show a distribution of tokens count per sentiment.
# - Add error bars or confidence intervals if you had a modeling task.

