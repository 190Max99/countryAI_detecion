# 农户环境照片 AI 积分制评分系统

本项目面向农村人居环境积分制评分场景，基于现场采集的农户环境照片，对室内、庭院、厕所、化粪池、房前屋后等场景中的扣分项进行自动识别，并按照人工评分细则生成扣分结果、场景得分和总分。

当前系统已经升级为一个可本地运行的完整评分闭环，支持：

- 五类场景照片的多标签扣分项识别；
- 根据人工评分细则自动计算场景得分和总分；
- 输入单个农户文件夹，一键输出人工分、预测分和分差；
- UI 界面选择文件夹并进行现场演示；
- 在原图上生成扣分项文字标注；
- 对模型关注区域生成 Grad-CAM 热力图（支持单标签和多标签模式）；
- 支持普通 ResNet18 模型和 ResNet18 + CBAM 注意力模型混合调用；
- 批量场景得分准确率 Excel 报告；
- 按扣分项标签维度的 AI vs 人工不一致分析；
- 训练过程曲线可视化；
- 支持后续阈值调整、模型替换和人工复核。

系统的核心思路不是让 AI 直接输出最终分数，而是先识别每张照片中是否存在具体扣分项，再根据人工评分标准计算得分。这样可以保证评分过程具有较好的可解释性，方便现场展示、人工复核和后续优化。

---

## 一、项目目标

本项目希望实现一个能够在本地电脑上运行的 AI 评分系统，完成从现场照片输入到自动评分输出的完整流程。

系统主要目标包括：

1. 对农户现场照片进行场景识别和扣分项识别；
2. 判断每个场景中是否存在对应扣分项；
3. 根据评分细则自动计算每个场景得分；
4. 汇总得到每户农户的 AI 预测总分；
5. 读取人工标注 CSV，计算人工实际总分；
6. 输出人工得分、模型预测得分和分差；
7. 在图片上生成扣分项文字标注，便于现场解释；
8. 基于 Grad-CAM 生成热力图，展示模型关注区域；
9. 提供 UI 界面，支持选择农户文件夹后自动评分；
10. 批量测评并输出 Excel 报告，便于模型迭代；
11. 作为农村人居环境现场评分的 AI 初评工具。

---

## 二、评分标准

系统按照"AI 积分制现场照片采样及评分细则"进行评分，总分为 60 分，分为四个主要部分：

| 类别           |  满分 | 说明                                                         |
| -------------- | ----: | ------------------------------------------------------------ |
| 室内           | 10 分 | 判断室内格局、家具、生活用品、鸡鸭共居、垃圾、墙面污迹等情况 |
| 庭院           | 30 分 | 判断生产工具、交通用具、鸡鸭、垃圾、柴草、污水、棚库等情况   |
| 厕所及化粪池   | 10 分 | 判断厕屋脏乱、功能配备、化粪池盖板、粪污溢流等情况           |
| 房前屋后及两侧 | 10 分 | 判断柴草堆放、污水横流、棚架破败、鸡鸭棚圈等情况             |

总分计算方式：

```text
总分 = 室内得分 + 庭院得分 + 厕所及化粪池得分 + 房前屋后得分
```

即：

```text
总分 = 60 - 所有扣分项总和
```

其中：

```text
场景得分 = 场景满分 - 该场景命中扣分项总扣分
```

厕所和化粪池需要合并计算：

```text
厕所及化粪池得分 = 10 - 厕所扣分 - 化粪池扣分
```

---

## 三、底层原理

### 3.1 从人工评分规则到 AI 标签

原始评分表中的每一个扣分项，都被转化为一个机器学习标签。

例如室内场景共有 A0 到 A9 共 10 个扣分项：

| 标签 | 含义                       | 扣分 |
| ---- | -------------------------- | ---: |
| A0   | 室内格局杂乱无章           |    1 |
| A1   | 室内家具摆放杂乱无章       |    1 |
| A2   | 室内生活用品杂乱无章       |    1 |
| A3   | 鸡鸭进入屋内共居           |    1 |
| A4   | 室内地面存在鸡鸭粪污       |    1 |
| A5   | 鸡跳在室内桌子上           |    1 |
| A6   | 室内地面垃圾乱丢现象严重   |    1 |
| A7   | 室内桌面、沙发表面乱堆乱摆 |    1 |
| A8   | 室内墙面污迹不堪           |    1 |
| A9   | 室内其他脏乱情况           |    1 |

一张图片可能同时存在多个问题，所以本项目不是普通单分类任务，而是多标签分类任务。

例如一张室内图片的人工标签可能是：

```text
A0 = 1, A1 = 1, A2 = 1, A3 = 0, A4 = 0, A5 = 0, A6 = 1, A7 = 1, A8 = 1, A9 = 0
```

对应到数据表中就是：

```text
label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9
1,1,1,0,0,0,1,1,1,0
```

其中 `1` 表示该项扣分，`0` 表示该项不扣分。

---

### 3.2 为什么使用多标签分类

普通图像分类通常是一张图片只属于一个类别，但本项目中一张图片可能同时存在多个扣分项。例如一张庭院照片可能同时存在 B1（交通用具杂乱）、B5（地面垃圾）、B10（污水横流），因此模型不能只输出一个类别，而是需要输出多个标签的概率。

模型输出形式类似：

```text
B0: 0.41, B1: 0.78, B2: 0.52, B3: 0.10, B4: 0.42, B5: 0.71, ...
```

系统再根据阈值判断是否扣分：

```text
概率 >= 阈值：判定该扣分项存在
概率 < 阈值：判定该扣分项不存在
```

---

### 3.3 ResNet18 模型原理

本项目使用 ResNet18 作为基础图像识别模型。ResNet18 可以理解为一个图像特征提取器，从图片中逐层提取视觉信息：

```text
原始图片 → 边缘/颜色/纹理 → 局部结构 → 物体特征 → 场景特征 → 输出多个扣分项概率
```

ResNet18 的核心是残差连接（Residual Connection）。普通卷积网络层数变深后容易出现梯度消失或训练退化问题，ResNet 通过短接结构将输入特征直接与卷积后的残差特征相加，使网络更容易训练。

---

### 3.4 迁移学习

由于项目数据量较少，不适合从零开始训练深度神经网络，因此采用迁移学习方法：

```text
使用在大规模图像数据（ImageNet）上预训练过的 ResNet18
保留其已有的基础视觉识别能力
只修改最后的全连接分类层，适配本项目的扣分标签数量
```

各场景模型需要修改的输出标签数：

| 模型     | 标签数 |
| -------- | -----: |
| 室内     |     10 |
| 庭院     |     12 |
| 厕所     |      2 |
| 化粪池   |      3 |
| 房前屋后 |      5 |

---

### 3.5 ResNet18 + CBAM 注意力机制

庭院场景内容复杂，包含生产工具、交通工具、柴草、垃圾、鸡鸭、污水、棚架等多种目标，且很多扣分项只占图像局部区域。普通 ResNet18 容易受到复杂背景干扰，因此系统对庭院模型增加了 CBAM（Convolutional Block Attention Module）注意力模块。

CBAM 由两部分组成：

- **Channel Attention（通道注意力）**：判断哪些特征通道更重要
- **Spatial Attention（空间注意力）**：判断图像哪些区域更重要

庭院 CBAM 模型流程为：

```text
庭院图片 → ResNet18 提取基础特征 → CBAM 增强关键通道和区域 → 全局平均池化 → 多标签分类头 → 输出扣分项概率
```

CBAM 主要用于提升庭院场景中局部扣分项的识别能力，例如地面垃圾、污水横流、柴草堆码、破败棚架、杂物堆放等。

> **注意**：CBAM 模型和普通 ResNet18 模型结构不同，加载时必须使用匹配的网络结构。当前系统已支持自动检测模型参数中是否包含 `cbam.` 前缀，并用对应的模型结构加载。CBAM 模块统一由 `src/cbam_resnet.py` 定义和维护。

---

### 3.6 Grad-CAM 热力图

系统支持 Grad-CAM 热力图功能，用于解释模型在判断某个扣分项时主要关注了图像中的哪些区域。热力图中红色越明显，表示模型在判断该扣分项时越关注该区域。

Grad-CAM 的作用流程：

```text
输入图片 → 选择某个扣分项输出 → 反向计算该扣分项对最后卷积特征图的贡献 → 生成热力图 → 叠加到原图上
```

系统当前支持两种 Grad-CAM 模式：

- **单标签模式**（`ui_folder_score_annotated_gradcam.py`）：仅对模型判定扣分的第一个标签生成热力图
- **多标签模式**（`ui_folder_score_annotated_gradcam_multi.py`）：对模型判定扣分的所有标签分别生成热力图

> **注意**：Grad-CAM 热力图展示的是模型关注区域，不是人工标注框或目标检测框，不能等同于物体精确位置。如需精确框出垃圾、污水、柴草等物体，需要进一步训练 YOLO 等目标检测模型。

---

### 3.7 为什么不直接预测最终分数

本项目没有采用"输入图片后直接输出分数"的方式，而是采用"先识别扣分项，再根据规则计算分数"。这种方式更适合现场应用，主要有三个优点：

1. **可解释性更强**：系统可以说明模型为什么扣分
2. **便于人工复核**：工作人员可查看模型识别出的扣分项、文字标注图和热力图
3. **便于后续维护**：评分规则变化时只需修改扣分权重或标签配置，不一定需要重新设计整个模型

---

## 四、模型训练过程

### 4.1 训练数据来源

训练数据来自 `data/all_labels.csv`，每一行包含 `house_id`、`scene`、`image_path` 和 `label_0` ~ `label_11`。对于某个场景模型，程序会先筛选对应场景的数据再训练。

### 4.2 训练流程

```text
读取 all_labels.csv → 筛选场景数据 → 读取图片和标签 → 预处理和数据增强
→ 输入 ResNet18 或 CBAM 模型 → 多标签分类 → BCEWithLogitsLoss 计算损失
→ 反向传播更新参数 → 保存 .pth 模型文件
```

使用的损失函数为 `BCEWithLogitsLoss`，因为每个标签本质上都是一个二分类任务（该扣分项是否存在）。

### 4.3 数据增强与模型保存

训练中每轮会进行随机裁剪、翻转、旋转、亮度变化等数据增强。训练完成后保存 `.pth` 文件，其中包含：

| 字段               | 含义                         |
| ------------------ | ---------------------------- |
| `model_state_dict` | 模型参数                     |
| `label_names`      | 标签名称                     |
| `label_cols`       | 标签列名                     |
| `deducts`          | 扣分权重                     |
| `thresholds`       | 预测阈值                     |
| `train_house_ids`  | 训练使用的农户编号           |
| `model_type`       | 模型类型（如 resnet18_cbam） |

---

## 五、现场应用流程

现场应用时，每户农户建议单独建立一个文件夹。例如第 97 户：

```text
data/raw/97/
├── 97.csv                  # 该户人工标注 CSV
├── 室内_97.jpg             # 室内照片
├── 庭院_97.jpg             # 庭院照片
├── 厕所_97.jpg             # 厕所照片
├── 化粪池_97.jpg           # 化粪池照片
└── 房前屋后_97.jpg         # 房前屋后照片
```

现场演示流程：

```text
打开 UI 界面 → 选择农户文件夹 → 点击开始评分
→ 系统自动读取 CSV 计算人工实际得分
→ 系统自动读取图片进行 AI 预测
→ 输出每个场景实际得分、预测得分、分差
→ 输出总分对比
→ 生成带扣分项文字标注的图片
→ 生成 Grad-CAM 热力图（支持单标签/多标签模式）
```

---

## 六、项目结构

```text
countryside_score/
│
├── src/                                          # 核心代码
│   ├── cbam_resnet.py                            # CBAM 注意力模块（通道+空间注意力）
│   │
│   ├── train_test_indoor.py                      # 室内模型训练与测试
│   ├── train_outside_70.py                       # 房前屋后模型训练
│   ├── train_outside_70_with_curves.py           # 房前屋后模型训练（带训练曲线可视化）
│   ├── train_courtyard_70.py                     # 普通庭院 ResNet18 训练
│   ├── train_courtyard_cbam_70.py                # 庭院 ResNet18 + CBAM 训练
│   ├── train_toilet_70.py                        # 厕所模型训练
│   ├── train_septic_70.py                        # 化粪池模型训练
│   │
│   ├── eval_indoor_current.py                    # 室内模型全量验证
│   ├── eval_outside_holdout.py                   # 房前屋后保留集测评
│   ├── eval_outside_current.py                   # 房前屋后当前模型整体测评
│   ├── eval_scene_score_accuracy_excel.py        # 批量场景得分准确率 Excel 报告
│   ├── eval_label_mismatch_by_scene_excel.py     # 按扣分项标签分析 AI vs 人工不一致
│   ├── plot_score_error_curves.py                # 得分误差曲线图绘制
│   │
│   ├── predict_indoor.py                         # 单张室内图片预测
│   ├── predict_outside.py                        # 单张房前屋后图片预测
│   ├── predict_septic.py                         # 单张化粪池图片预测
│   ├── predict_house_total.py                    # 根据 house_id 进行整户预测
│   ├── predict_folder_total.py                   # 输入文件夹进行整户评分
│   │
│   ├── ui_folder_score.py                        # 基础 UI 界面
│   ├── ui_folder_score_annotated.py              # 带文字标注的 UI 界面
│   ├── ui_folder_score_annotated_cbam_fixed.py   # 支持庭院 CBAM 模型的 UI 界面
│   ├── ui_folder_score_annotated_gradcam.py      # 支持 Grad-CAM 热力图（单标签）的 UI
│   └── ui_folder_score_annotated_gradcam_multi.py# 支持 Grad-CAM 热力图（多标签）的 UI
│
├── scripts/                                      # 数据整理与工具脚本
│   └── build_dataset.py                          # 将每户 CSV 整理为 all_labels.csv
│
├── data/                                         # 数据目录（本地保存，不上传 Git）
│   ├── raw/                                      # 原始农户照片与每户标注 CSV
│   │   ├── 1/  ~  100/                           # 各农户文件夹
│   │   └── ...
│   ├── all_labels.csv                            # 整理后的总标注表
│   └── all_score.csv                             # 总得分表
│
├── models/                                       # 训练好的模型文件（本地保存，不上传 Git）
│   ├── indoor_resnet18.pth                       # 室内模型（10 标签）
│   ├── courtyard_resnet18.pth                    # 普通庭院模型（12 标签）
│   ├── courtyard_resnet18_cbam.pth               # 庭院 CBAM 模型（12 标签）
│   ├── toilet_resnet18.pth                       # 厕所模型（2 标签）
│   ├── septic_resnet18.pth                       # 化粪池模型（3 标签）
│   └── outside_resnet18.pth                      # 房前屋后模型（5 标签）
│
├── outputs/                                      # 测评结果与报告输出
│   ├── training_curves/                          # 训练过程曲线图
│   ├── scene_score_accuracy_report.xlsx          # 场景得分准确率报告
│   ├── label_mismatch_report.xlsx                # 扣分标签不一致分析报告
│   ├── outside_training_history.csv              # 房前屋后训练历史
│   └── ...
│
├── docs/                                         # 文档目录
│
├── test/                                         # 测试目录
│
├── predict_outside_flowchart.md                  # predict_outside.py 流程图文档
├── train_courtyard_cbam_flowchart.md             # 庭院 CBAM 训练流程图文档
├── train_outside_70_flowchart.md                 # 房前屋后训练流程图文档
├── ui_folder_score_flowchart.md                  # UI 评分流程图文档
│
├── app.py                                        # 应用入口（待扩展）
├── config.py                                     # 场景与标签集中配置
├── check_csv.py                                  # CSV 编码检测工具
├── requirements.txt                              # Python 依赖
├── .gitignore                                    # Git 忽略规则
└── README.md                                     # 项目说明文档（本文件）
```

**目录说明：**

| 目录/文件        | 用途                                       |
| ---------------- | ------------------------------------------ |
| `src/`           | 存放训练、预测、测评、UI 等核心代码        |
| `scripts/`       | 数据整理与工具脚本                         |
| `data/raw/`      | 原始图片和每户人工标注 CSV                 |
| `models/`        | 训练好的`.pth` 模型文件                    |
| `outputs/`       | 批量测评结果、训练曲线、Excel 报告         |
| `docs/`          | 补充文档                                   |
| `*.md`（根目录） | 各核心模块的业务流程图文档                 |
| `config.py`      | 场景配置（标签名、扣分值、阈值、模型路径） |
| `check_csv.py`   | 快速检测`all_labels.csv` 编码和结构        |

---

## 七、环境安装

建议使用 Python 虚拟环境运行项目。

```powershell
# 进入项目目录
cd D:\desktop\countryside_score

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

**依赖清单（`requirements.txt`）：**

| 包名         | 用途                                |
| ------------ | ----------------------------------- |
| torch        | 深度学习框架                        |
| torchvision  | 图像模型与预处理                    |
| pandas       | 数据处理与 CSV 读写                 |
| numpy        | 数值计算                            |
| pillow       | 图像读写与标注绘制                  |
| scikit-learn | 评估指标（F1 / Precision / Recall） |
| tqdm         | 训练进度条                          |
| matplotlib   | 训练曲线与热力图绘制                |
| openpyxl     | Excel 报告生成（测评脚本需要）      |

`tkinter` 一般随 Python 自带，无需单独安装。

---

## 八、数据格式

### 8.1 总标注文件（`data/all_labels.csv`）

```csv
house_id,scene,image_path,label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11
1,室内,data/raw/1/室内_1.jpg,1,0,1,0,0,0,0,0,0,0,,
1,庭院,data/raw/1/庭院_1.jpg,1,1,1,0,0,0,0,0,0,0,0,0
1,房前屋后,data/raw/1/房前屋后_1.jpg,0,0,0,0,0,,,,,,,
```

| 字段                   | 含义                                               |
| ---------------------- | -------------------------------------------------- |
| `house_id`             | 农户编号                                           |
| `scene`                | 场景名称（室内 / 庭院 / 厕所 / 化粪池 / 房前屋后） |
| `image_path`           | 图片路径（相对于项目根目录）                       |
| `label_0` ~ `label_11` | 扣分标签，1 = 扣分，0 = 不扣分，空 = 不适用        |

### 8.2 每户文件夹内 CSV 格式

用于 UI 演示时，每户文件夹中放置单户 CSV（如 `data/raw/97/97.csv`）：

```csv
序号,label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11
厕所,1,1,,,,,,,,,,
房前屋后,0,0,0,0,1,,,,,,,
室内,1,1,1,0,0,0,1,1,1,0,,
庭院,1,0,0,0,0,0,0,0,0,0,0,0
化粪池,0,0,1,,,,,,,,,
```

该 CSV 用于计算人工实际得分。系统支持多种编码自动检测（UTF-8-SIG / GBK / GB18030 / UTF-8）。

---

## 九、训练各场景模型

> 训练和预测建议始终在项目根目录运行，使用 `python -m src.xxx` 方式调用。

### 9.1 室内模型训练

室内场景对应 A0 ~ A9，共 10 个标签。

```powershell
python -m src.train_test_indoor --csv data/all_labels.csv --epochs 30
```

输出：`models/indoor_resnet18.pth`、`indoor_test_result.csv`

---

### 9.2 房前屋后模型训练

房前屋后场景对应 D0 ~ D4，共 5 个标签。

**基础训练：**

```powershell
python -m src.train_outside_70 --csv data/all_labels.csv --epochs 30 --train_num 70
```

**带训练曲线的训练（推荐）：**

```powershell
python -m src.train_outside_70_with_curves --csv data/all_labels.csv --epochs 40 --train_num 70
```

带曲线版本会在训练过程中记录每个 epoch 的训练 Loss、验证 Loss、标签准确率、整图一致率、评分 MAE 和分差，训练结束后自动保存历史 CSV 和多张曲线图到 `outputs/training_curves/`。

输出：`models/outside_resnet18.pth`、`outside_holdout_rows.csv`

---

### 9.3 普通庭院模型训练

庭院场景对应 B0 ~ B11，共 12 个标签。

```powershell
python -m src.train_courtyard_70 --csv data/all_labels.csv --epochs 30 --train_num 70
```

输出：`models/courtyard_resnet18.pth`、`courtyard_holdout_rows.csv`

---

### 9.4 庭院 ResNet18 + CBAM 模型训练

为提升复杂庭院场景下局部扣分项识别效果，支持训练带 CBAM 注意力模块的庭院模型。

```powershell
python -m src.train_courtyard_cbam_70 --csv data/all_labels.csv --epochs 40 --train_num 70
```

输出：`models/courtyard_resnet18_cbam.pth`、`courtyard_cbam_holdout_rows.csv`

---

### 9.5 厕所模型训练

厕所模型对应 2 个标签：C0（厕屋脏乱）、C1（厕屋功能配备不齐全）。

```powershell
python -m src.train_toilet_70 --csv data/all_labels.csv --epochs 30 --train_num 70
```

输出：`models/toilet_resnet18.pth`、`toilet_holdout_rows.csv`

---

### 9.6 化粪池模型训练

化粪池模型对应 3 个标签：化粪池盖板未关闭、粪污溢流、厕所周围其他情况。

```powershell
python -m src.train_septic_70 --csv data/all_labels.csv --epochs 30 --train_num 70
```

输出：`models/septic_resnet18.pth`、`septic_holdout_rows.csv`

---

## 十、单张图片预测

### 10.1 单张室内图片预测

```powershell
python -m src.predict_indoor --image "data/raw/55/室内_55.jpg"
```

输出：A0 ~ A9 每个标签的预测概率、是否扣分、室内总扣分、室内最终得分。

### 10.2 单张房前屋后图片预测

```powershell
python -m src.predict_outside --image "data/raw/71/房前屋后_71.jpg"
```

输出：D0 ~ D4 每个标签的预测概率、是否扣分、房前屋后总扣分、最终得分。

### 10.3 单张化粪池图片预测

```powershell
python -m src.predict_septic --image "data/raw/61/化粪池_61.jpg"
```

输出：化粪池各标签预测概率、是否扣分、化粪池部分扣分。

> 注意：化粪池不单独计 10 分，需与厕所扣分合并：`厕所及化粪池得分 = 10 - 厕所扣分 - 化粪池扣分`。

---

## 十一、输入文件夹进行整户评分

直接输入一个农户文件夹进行整体测评：

```powershell
python -m src.predict_folder_total --folder data/raw/97
```

该命令自动完成：读取 CSV → 读取各场景图片 → 计算人工得分 → AI 预测 → 输出各场景对比。

**输出示例：**

```text
========== 评分结果 ==========
      项目  满分  实际扣分  预测扣分  实际得分  预测得分  分差
      室内  10      6      4      4      6    2
      庭院  30      3      8     27     22   -5
厕所及化粪池 10      6      6      4      4    0
  房前屋后 10      1      1      9      9    0
      总分  60     16     19     44     41   -3
```

分差 = 预测得分 − 实际得分。分差 > 0 表示 AI 给分偏高，分差 < 0 表示 AI 给分偏低。

---

## 十二、UI 界面运行

系统提供多个版本的 UI 界面，推荐使用功能最全的 Grad-CAM 多标签版本：

```powershell
# 基础版 UI
python -m src.ui_folder_score

# 带扣分项文字标注的 UI
python -m src.ui_folder_score_annotated

# 支持庭院 CBAM 模型的 UI
python -m src.ui_folder_score_annotated_cbam_fixed

# 支持 Grad-CAM 热力图（单标签）的 UI
python -m src.ui_folder_score_annotated_gradcam

# 支持 Grad-CAM 热力图（多标签，推荐使用）的 UI
python -m src.ui_folder_score_annotated_gradcam_multi
```

**操作步骤：**

1. 点击"选择农户文件夹"
2. 选择 `data/raw/97` 这类文件夹
3. 点击"开始评分"
4. 查看评分结果表格
5. 查看 `annotated/` 中的文字标注图
6. 查看 `gradcam/` 中的热力图

UI 运行后会在农户文件夹下生成：

```text
data/raw/97/annotated/     # 带扣分项文字标注的图片
data/raw/97/gradcam/       # Grad-CAM 热力图
```

---

## 十三、图片扣分项文字标注

UI 的标注版本支持在图片上自动写出模型预测出的扣分项。例如庭院图片被模型判定存在 B1（交通用具杂乱）和 B5（地面垃圾），系统会在图片上生成文字标注：

```text
场景：庭院
模型扣分：8
扣分项：
B1_庭院内交通用具杂乱无章 | 概率 0.778 | -3 分
B5_庭院内地面垃圾乱丢现象严重 | 概率 0.716 | -5 分
```

生成结果保存在 `data/raw/{house_id}/annotated/` 目录中。

---

## 十四、Grad-CAM 热力图

系统支持生成 Grad-CAM 热力图，展示模型在判断每个扣分项时关注的图像区域。

- **单标签模式**：仅对第一个被判定扣分的标签生成热力图
- **多标签模式**（推荐）：对所有被判定扣分的标签分别生成热力图

运行 UI 后会在 `data/raw/{house_id}/gradcam/` 中生成文件，例如：

```text
庭院_97_B5_庭院内地面垃圾乱丢现象严重_热力图.jpg
室内_97_A6_室内地面垃圾乱丢现象严重_热力图.jpg
```

> Grad-CAM 是模型关注区域可视化，不是目标检测框。如需精确框出物体位置，需要引入 YOLO 等目标检测模型。

---

## 十五、模型测评与报告

### 15.1 室内模型全量验证

```powershell
python -m src.eval_indoor_current --csv data/all_labels.csv --model models/indoor_resnet18.pth
```

输出指标：标签平均准确率、整图标签完全一致率、Macro F1 / Precision / Recall、平均得分误差、各标签 F1。

---

### 15.2 房前屋后模型测评

**保留集测评：**

```powershell
python -m src.eval_outside_holdout --holdout outside_holdout_rows.csv --model models/outside_resnet18.pth
```

**全量当前数据测评：**

```powershell
python -m src.eval_outside_current --csv data/all_labels.csv --model models/outside_resnet18.pth
```

---

### 15.3 批量场景得分准确率 Excel 报告（🆕）

自动扫描 `data/raw` 下所有农户文件夹，对每户进行评分并生成综合 Excel 报告：

```powershell
python -m src.eval_scene_score_accuracy_excel --root data/raw --output outputs/scene_score_accuracy_report.xlsx
```

报告包含：每户得分明细、每张图片明细、场景准确率汇总（含 1 分内 / 2 分内准确率、MAE、RMSE）、单图扣分准确率、标签概率明细。

---

### 15.4 扣分项标签维度 AI vs 人工不一致分析（🆕）

按每个扣分项标签，统计模型与人工标注的不一致情况：

```powershell
python -m src.eval_label_mismatch_by_scene_excel --root data/raw --output outputs/label_mismatch_report.xlsx
```

报告包含：每个扣分项的不一致率、误扣率、漏扣率、Precision、Recall、F1，便于分析"庭院 B5 地面垃圾"等具体问题的模型表现。

---

### 15.5 得分误差曲线图（🆕）

汇总 UI 评分结果，绘制得分误差分布曲线：

```powershell
python -m src.plot_score_error_curves --root data/raw --output outputs/score_error_curves.png
```

---

## 十六、阈值调整

模型输出的是概率，系统根据阈值判断是否扣分（概率 ≥ 阈值 → 扣分，概率 < 阈值 → 不扣分）。

**调整方向：**

| 现象                | 原因         | 调整方向 |
| ------------------- | ------------ | -------- |
| 预测得分 < 实际得分 | 模型扣分太多 | 阈值调高 |
| 预测得分 > 实际得分 | 模型扣分太少 | 阈值调低 |

例如庭院中 B5 是 5 分大项，误扣会导致分差很大，可适当提高 B5 阈值减少误扣。

各场景默认阈值配置在 `config.py` 的 `DEFAULT_THRESHOLDS` 字典中。

---

## 十七、流程图文档

项目提供了核心模块的业务流程图，便于理解代码逻辑：

| 文档                                | 对应模块                          |
| ----------------------------------- | --------------------------------- |
| `predict_outside_flowchart.md`      | 单张房前屋后图片预测流程          |
| `train_courtyard_cbam_flowchart.md` | 庭院 CBAM 模型训练流程            |
| `train_outside_70_flowchart.md`     | 房前屋后模型训练（70 组数据）流程 |
| `ui_folder_score_flowchart.md`      | UI 文件夹评分流程                 |

---

## 十八、Git 管理说明

**不上传 Git 的内容：**

```text
venv/               # 虚拟环境
data/raw/           # 原始图片和标注
models/             # 模型文件（.pth）
outputs/            # 输出报告
*.pth, *.pt, *.onnx # 模型权重
*.jpg, *.jpeg, *.png, *.bmp, *.webp  # 图片文件
*_result.csv, *_eval*.csv  # 临时结果
annotated/, gradcam/  # UI 生成输出
```

**建议上传 Git 的内容：**

```text
src/                # 核心代码
scripts/            # 工具脚本
README.md           # 项目文档
requirements.txt    # 依赖清单
.gitignore          # 忽略规则
config.py           # 配置文件
*.md                # 流程图文档
check_csv.py        # CSV 检测工具
```

---

## 十九、注意事项

1. 训练和预测建议始终在项目根目录运行，使用 `python -m src.xxx` 方式调用。
2. 图片路径应相对于项目根目录，例如 `data/raw/71/房前屋后_71.jpg`。
3. 每户文件夹中的 CSV 应包含"序号"或"scene"列，并包含 `label_0` 到对应标签列。
4. 数据量较少时，模型结果只能作为 AI 初评，建议保留人工复核。
5. 如果重新训练模型，默认会覆盖 `models/` 中同名模型文件。
6. 若更换电脑，需要重新创建虚拟环境并安装依赖。
7. 当前图片标注是扣分项文字标注，不是目标检测框。
8. Grad-CAM 热力图是模型关注区域，不是精确目标框。
9. 若需要检测具体物体位置，需后续引入 YOLO 等目标检测模型。
10. CBAM 模型和普通 ResNet18 结构不同，加载时必须使用匹配的网络结构。当前 UI 已支持自动判断。
11. 庭院、厕所、化粪池等场景中部分标签样本较少时，模型效果可能不稳定，需要继续补充数据或调整阈值。
12. `openpyxl` 是生成 Excel 报告的必要依赖，请确保已安装。

---

## 二十、项目总结

本项目将农村人居环境积分制评分规则转化为一个可运行的 AI 图像评分系统。系统通过多标签分类模型识别每张现场照片中的扣分项，再按照人工评分细则计算每个场景得分和总分。

当前系统已实现的完整闭环：

```text
现场照片输入
    ↓
扣分项自动识别（ResNet18 / ResNet18 + CBAM）
    ↓
场景得分与总分计算
    ↓
人工分与预测分对比、分差分析
    ↓
带扣分项文字标注图片输出
    ↓
Grad-CAM 热力图输出（单标签 / 多标签）
    ↓
批量 Excel 报告（得分准确率 + 标签不一致分析）
    ↓
UI 界面现场演示
```

本系统适合作为农村人居环境现场评分的 AI 初评工具，也可以作为后续目标检测、自动标注和智能巡检系统的基础版本。
