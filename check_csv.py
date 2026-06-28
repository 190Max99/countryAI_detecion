import pandas as pd
from pathlib import Path

csv_path = Path("data/all_labels.csv")

print("文件是否存在:", csv_path.exists())
print("文件绝对路径:", csv_path.resolve())

encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]

for enc in encodings:
    try:
        print("\n尝试编码:", enc)
        df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
        print("读取成功")
        print("列名:", df.columns.tolist())
        print(df.head())
        break
    except Exception as e:
        print("读取失败:", e)