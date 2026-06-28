# 农户环境照片 AI 积分制评分系统

本项目用于农村人居环境积分制评分场景。系统基于多标签图像分类模型，对农户现场照片中的扣分项进行自动识别，并根据评分细则生成扣分结果和得分。

当前项目主要实现了：

* 室内照片多标签识别与评分
* 房前屋后照片多标签识别与评分
* 本地训练 ResNet18 多标签分类模型
* 单张图片预测
* 批量测评模型效果
* 支持人工标注数据训练与后续验证

## 一、项目结构

```text
countryside_score/
│
├── src/                         # 核心代码
│   ├── train_test_indoor.py      # 室内模型训练与测试
│   ├── eval_indoor_current.py    # 室内模型重新测评
│   ├── predict_indoor.py         # 单张室内图片预测
│   ├── train_outside_70.py       # 房前屋后模型训练
│   ├── eval_outside_holdout.py   # 房前屋后保留集测评
│   └── predict_outside.py        # 单张房前屋后图片预测
│
├── scripts/                     # 数据整理脚本
│   └── build_dataset.py
│
├── data/                        # 数据目录，本地保存，不上传 Git
│   ├── raw/                     # 原始农户照片与标注文件
│   └── all_labels.csv           # 整理后的总标注表
│
├── models/                      # 训练好的模型文件，本地保存，不上传 Git
│   ├── indoor_resnet18.pth
│   └── outside_resnet18.pth
│
├── outputs/                     # 测评结果输出，本地保存
│
├── .gitignore                   # Git 忽略规则
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明
```

说明：

* `data/raw/` 中存放原始图片和每户标注文件；
* `models/` 中存放训练好的 `.pth` 模型；
* `data/`、`models/`、`venv/` 不建议上传到 Git；
* Git 只管理代码、配置文件和说明文档。

## 二、环境安装

建议使用虚拟环境运行项目。

```powershell
cd D:\desktop\countryside_score
python -m venv venv
venv\Scripts\activate
```

安装依赖：

```powershell
pip install -r requirements.txt
```

如果没有 `requirements.txt`，可以手动安装：

```powershell
pip install torch torchvision pandas numpy pillow scikit-learn tqdm matplotlib
```

## 三、数据格式

整理后的总标注文件为：

```text
data/all_labels.csv
```

基本格式如下：

```csv
house_id,scene,image_path,label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11
1,室内,data/raw/1/室内_1.jpg,1,0,1,0,0,0,0,0,0,0,,
1,庭院,data/raw/1/庭院_1.jpg,1,1,1,0,0,0,0,0,0,0,0,0
1,房前屋后,data/raw/1/房前屋后_1.jpg,0,0,0,0,0,,,,,,,
```

其中：

* `house_id`：农户编号；
* `scene`：照片场景，例如室内、庭院、厕所、化粪池、房前屋后；
* `image_path`：图片路径；
* `label_0` 到 `label_11`：对应场景下的扣分标签；
* `1` 表示该项扣分；
* `0` 表示该项不扣分。

## 四、室内模型训练

室内场景对应 `label_0` 到 `label_9`，即 A0 到 A9。

运行：

```powershell
python src/train_test_indoor.py --csv data/all_labels.csv --epochs 30
```

训练完成后会生成：

```text
models/indoor_resnet18.pth
indoor_test_result.csv
```

其中：

* `models/indoor_resnet18.pth` 是训练好的室内模型；
* `indoor_test_result.csv` 是测试集预测结果。

## 五、室内模型测评

对当前室内模型重新测评：

```powershell
python src/eval_indoor_current.py --csv data/all_labels.csv --model models/indoor_resnet18.pth
```

输出结果包括：

* 标签平均准确率；
* 整张图片标签完全一致率；
* Macro F1；
* Precision；
* Recall；
* 平均得分误差；
* 各标签 F1。

测评结果会保存为：

```text
indoor_eval_current_result.csv
```

## 六、单张室内图片预测

```powershell
python src/predict_indoor.py --image "data/raw/1/室内_1.jpg"
```

输出内容包括：

* A0 到 A9 每个标签的预测概率；
* 每个标签是否扣分；
* 总扣分；
* 室内最终得分。

## 七、房前屋后模型训练

房前屋后场景对应 `label_0` 到 `label_4`，即 D0 到 D4。

使用前 70 组数据训练，剩余数据保留用于验证：

```powershell
python src/train_outside_70.py --csv data/all_labels.csv --epochs 30
```

训练完成后会生成：

```text
models/outside_resnet18.pth
outside_holdout_rows.csv
```

其中：

* `models/outside_resnet18.pth` 是房前屋后模型；
* `outside_holdout_rows.csv` 是未参与训练的保留验证数据。

## 八、房前屋后模型测评

使用保留数据测评模型：

```powershell
python src/eval_outside_holdout.py --holdout outside_holdout_rows.csv --model models/outside_resnet18.pth
```

测评结果会保存为：

```text
outside_holdout_eval_result.csv
```

## 九、单张房前屋后图片预测

```powershell
python src/predict_outside.py --image "data/raw/71/房前屋后_71.jpg"
```

输出内容包括：

* D0 到 D4 每个标签的预测概率；
* 每个标签是否扣分；
* 总扣分；
* 房前屋后最终得分。

## 十、评分逻辑

本项目不直接让模型输出最终分数，而是采用：

```text
图片输入
↓
模型预测多个扣分标签
↓
根据评分细则计算扣分
↓
得到最终得分
```

例如室内照片：

```text
模型预测：A0=1，A2=1，其余为0
扣分：1 + 1 = 2
室内得分：10 - 2 = 8分
```

这种方式比直接预测分数更容易解释，也便于人工复核。

## 十一、Git 管理说明

本项目使用 Git 管理代码，但不管理大文件。

不上传 Git 的内容包括：

```text
venv/
data/raw/
models/
outputs/
*.pth
*.jpg
*_result.csv
```

这些内容只保存在本地。

推荐提交到 Git 的内容包括：

```text
src/
scripts/
README.md
requirements.txt
.gitignore
config.py
```

初始化提交示例：

```powershell
git add .gitignore README.md requirements.txt src/ scripts/ config.py
git commit -m "初始化农户AI评分项目代码"
```

## 十二、注意事项

1. 训练和预测建议始终在项目根目录运行，不要进入 `src/` 后再运行脚本。
2. 图片路径应相对于项目根目录，例如 `data/raw/71/房前屋后_71.jpg`。
3. 数据量较少时，模型结果只能作为 AI 初评，建议保留人工复核。
4. 如果重新训练模型，默认会覆盖 `models/` 中同名模型文件。
5. 若更换电脑，需要重新创建虚拟环境并安装依赖。
