import os
import glob
import matplotlib.pyplot as plt
from tbparse import SummaryReader

# 指定 TensorBoard 日志目录，里面应该有 events.out.tfevents.* 文件
log_dir = "logs"  
output_dir = "visualization_model"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用 tbparse 读取日志文件
reader = SummaryReader(log_dir, pivot=True)
# reader.scalars 会返回一个包含所有标量数据的 DataFrame

df = reader.scalars
if df.empty:
    print("没有在指定目录下找到任何可用的标量数据！")
else:
    # 按照 columns 列表中的标签绘制图表, 除了 `step`
    columns = df.columns.tolist()

    for column in columns:
        if column == "step":
            continue
        plt.figure(figsize=(10, 6))
        plt.plot(df["step"], df[column], label=column)
        column = column.replace('/', '_')
        plt.xlabel("Step")
        plt.ylabel(column)
        plt.title(f'{column} vs. Step')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{column}.png", dpi=500)
        plt.close()