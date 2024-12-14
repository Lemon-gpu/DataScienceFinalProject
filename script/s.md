### 封面页 (Cover Slide)

尊敬的各位老师、同学，大家好！
Hello everyone, esteemed professors and fellow students!

欢迎来到我的演讲，主题是基于机器学习的金融文本情感分析。
Welcome to my presentation on Machine Learning Based Sentiment Analysis of Financial Texts.

我是澳门科技大学的贾川、李博和苏晟。
I am Chuan Jia, Bo Li, and Sheng Su from Macau University of Science and Technology.

---

### 问题陈述 (Problem Statement)

金融领域中大量的文本数据，如新闻、报告和社交媒体，都会影响投资决策。
Massive volumes of financial text, such as news, reports, and social media, influence investment decisions.

然而，传统的分析方法往往忽视了文本数据中细微的情感线索。
However, traditional analysis often overlooks subtle emotional cues embedded in textual data.

我们的目标是将情感分析整合到金融决策中，以增强其合理性和预测准确性。
Our objective is to integrate sentiment analysis into financial decision-making to enhance rationality and predictive accuracy.

---

### 研究目标 (Objectives)

当前的挑战是缺乏高质量、注释完善的中文金融情感语料库。
The challenge is the limited availability of high-quality, annotated Chinese financial sentiment corpora.

因此，我们需要一种稳健的方法来生成和利用特定领域的标注数据集。
Hence, there is a need for a robust methodology to generate and leverage domain-specific labeled datasets.

最终目标是通过数据增强和基于Transformer的模型，提高中文金融情感分类的准确性。
The goal is to improve Chinese financial sentiment classification via data augmentation and Transformer-based models.

---

### 数据集描述 (Dataset Description)

我们的数据集包括来自多种来源的扩展中文金融新闻文章。
Our dataset consists of extended Chinese financial news articles from various sources.

情感标签分为积极、中立和消极三类。
Sentiment labels are categorized into Positive, Neutral, and Negative.

数据集规模为54,749篇已标注的文章。
The dataset size comprises 54,749 labeled articles.

---

### 数据预处理 (Data Preprocessing)

我们将高质量的英文金融情感语料库翻译成中文，并使用领域知识来标注数据。
We translated high-quality labeled English financial sentiment corpora into Chinese, labeling data with domain knowledge.

这么做很好的扩充了中文情感数据集。
This effectively expanded the Chinese sentiment dataset.

对于未标注的数据，我们使用中文金融情感词典进行预标注。
For non-labeled data, we used a Chinese financial sentiment lexicon for prelabeling.

文本分词采用了结巴中文分词工具。
Tokenization was performed using the Jieba Chinese tokenizer.

---

### 数据集可视化 (Dataset Visualization)

这是我们数据集中情感分布的图示。
This is a visualization of sentiment distribution in our dataset.

可以看到，积极、消极和中立情感的比例。
You can observe the proportions of Positive, Negative, and Neutral sentiments.

我们可以看到，本数据集的分布其实是不平衡的，积极的情感标签远多于剩余的两种情感标签。
We can see that the distribution of this dataset is actually imbalanced, with Positive sentiment labels far outnumbering the other two.

---

### 中立情感词云 (Dataset Visualization: Neutral Sentiment)

这是中立情感的词云图。
This is a word cloud for Neutral Sentiment.

图中展示了在中立情感中高频出现的词汇。
The figure displays high-frequency words appearing in Neutral Sentiment.

---

### 积极情感词云 (Dataset Visualization: Positive Sentiment)

这是积极情感的词云图。
This is a word cloud for Positive Sentiment.

图中展示了在积极情感中高频出现的词汇。
The figure displays high-frequency words appearing in Positive Sentiment.

---

### 消极情感词云 (Dataset Visualization: Negative Sentiment)

这是消极情感的词云图。
This is a word cloud for Negative Sentiment.

图中展示了在消极情感中高频出现的词汇。
The figure displays high-frequency words appearing in Negative Sentiment.

---

### 方法概述 (Methods Overview)

首先，数据准备与增强：
Firstly, Data Preparation and Augmentation:

我们使用基于Transformer的神经机器翻译模型，将高质量的英文金融情感语料库翻译成中文。
We used a Transformer-based Neural Machine Translation (TNMT) model to translate high-quality English financial sentiment corpora into Chinese.

其次，基于词典的标注：
Secondly, Lexicon-Based Annotation:

我们采用中文金融情感词典进行标注，参考了Jiang等人的研究。
We used a Chinese financial sentiment lexicon for labeling, inspired by Jiang et al. (2019).

通过领域特定术语，分配积极、中立和消极情感。
Sentiments (positive, neutral, negative) were assigned via domain-specific terms.

这样，我们扩充了用于情感分析的中文语料库。
As a result, we enlarged the Chinese corpus for sentiment analysis.

最终，我们使用标注的数据集对预训练的中文BERT模型进行微调。
Finally, we fine-tuned a pre-trained Chinese BERT model with the annotated dataset.

---

### 结果：训练损失 (TNMT) (Results: Training Loss (TNMT))

这是Transformer模型在训练过程中的损失曲线。
This is the training loss curve of the Transformer model.

在200步后，损失趋于稳定，达到了6.55。
After 200 steps, the loss stabilized at 6.55.

---

### 结果：BLEU得分 (TNMT) (Results: BLEU Score (TNMT))

这是Transformer模型在评估过程中的BLEU得分。
This is the BLEU score of the Transformer model during evaluation.

BLEU得分是机器翻译质量的评估指标，这对于我们的任务非常重要。
The BLEU score is an evaluation metric for machine translation quality, crucial for our task.

在130步后，BLEU得分稳定在0.08。
After 130 steps, the BLEU score stabilized at 0.08.

---

### 结果：F1得分 (TNMT) (Results: F1 Score (TNMT))

这是Transformer模型在评估过程中的F1得分。
This is the F1 score of the Transformer model during evaluation.

F1得分是一种综合评估指标，用于衡量模型的准确性。由于我们的数据并不平衡，F1得分对于我们的任务非常重要。
The F1 score is a comprehensive evaluation metric to measure model accuracy, crucial for our imbalanced data.

在200步后，F1得分稳定在0.0015。
After 200 steps, the F1 score stabilized at 0.0015.

---

### 结果：训练准确率 (BERT) (Results: Training Accuracy (BERT))

这是BERT模型在训练过程中的准确率曲线。
This is the training accuracy curve of the BERT model.

准确率不断提高，最终达到0.46。
Accuracy continuously increased, ending at 0.46.

---

### 结果：训练F1得分 (BERT) (Results: Training F1 Score (BERT))

这是BERT模型在训练过程中的F1得分曲线。
This is the training F1 score curve of the BERT model.

在60步后，F1得分稳定在0.39。
After 60 steps, the F1 score stabilized at 0.39.

---

### 结果：训练损失 (BERT) (Results: Training Loss (BERT))

这是BERT模型在训练过程中的损失曲线。
This is the training loss curve of the BERT model.

在120步后，损失稳定在1.06。
After 120 steps, the loss stabilized at 1.06.

---

### 结果 - 领域适应 (Results - Domain Adaptation)

我们的研究表明，我们的方法具有潜在的领域适应能力。
Our research indicates the potential for domain adaptation with our approach.

这对于增强动态金融环境下模型的鲁棒性有一定的启发性。
This is insightful for enhancing model robustness in dynamic financial environments.

同时，这种方法在其他语言和领域中的可扩展性也是有待研究的。
Moreover, the scalability of this approach to other languages and domains is worth exploring.

我们的模型的演示如下：
Here is a demonstration of our model:

中文文本："在英国央行本次议息会议上，市场焦点可能会落在表决结果和政策制定者的沟通上。如果投票结果显示是一个势均力敌的决定，并且会议声明再次指出不急于进一步降息，可能打压未来的宽松预期。"
Chinese Text: "In the recent meeting of the Bank of England, the market focus is likely to be on the voting results and communication from policymakers. If the vote results show a deadlock and the meeting statement again points out no rush to further rate cuts, it may dampen future easing expectations."

情感标签：消极
Sentiment Label: Negative

---

### 结论：贡献与背景 (Conclusion: Contributions & Background)

我们的主要贡献包括：
Our main contributions include:

使用TNMT创建了中文情感数据。
Using TNMT to create Chinese sentiment data.

应用基于词典的标注方法扩展了数据集。
Applying lexicon-based annotation to expand datasets.

微调BERT模型以提高分类性能。
Fine-tuning the BERT model for better classification.

在金融文本方面，包含市场分析、社交媒体和报告。
Regarding financial texts, including market analyses, social media, and reports.

情感信息对于情感、趋势和机会至关重要。
Emotional information is key for sentiment, trends, and opportunities.

面临的挑战是手动分析速度太慢。
The challenge is that manual analysis is too slow.

我们的解决方案是潜在的自动化系统。
Our solution is potential automated systems.

这样，我们可以更快地做出决策。
Thus, we can make decisions faster.

---

### 结论：改进与未来方向 (Conclusion: Improvements and Future Directions)

当前系统的局限性包括：
Current system limitations include:

翻译后的质量问题。
Post-translation quality issues.

词典信息的利用不充分。
Ineffective use of dictionary information.

需要更好的数据过滤和质量控制。
Need for better data filtering and quality control.

未来的研究方向：
Future directions:

细化情感分析模块。
Fine-tuning emotional analysis modules.

利用知识图谱进行语义对齐。
Leveraging knowledge graphs for semantic alignment.

探索扩散模型用于情感预测。
Exploring diffusion models for sentiment prediction.

---

### 结束语

感谢大家的聆听！
Thank you all for listening!

代码资源可在[GitHub链接](https://github.com/Lemon-gpu/DataScienceFinalProject)找到。
Code resources can be found at [GitHub link](https://github.com/Lemon-gpu/DataScienceFinalProject).

视频资源将上传至YouTube。
Video resources will be uploaded to YouTube.
