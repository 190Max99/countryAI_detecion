# 农户环境照片 AI 积分制评分系统

本项目面向农村人居环境积分制评分场景，基于现场采集的农户环境照片，对室内、庭院、厕所、化粪池、房前屋后等场景中的扣分项进行自动识别，并按照人工评分细则生成扣分结果、场景得分和总分。

当前系统已经从最初的“单场景识别与评分”升级为一个可本地运行的完整评分闭环，支持：

- 五类场景照片的多标签扣分项识别；
- 根据人工评分细则自动计算场景得分和总分；
- 输入单个农户文件夹，一键输出人工分、预测分和分差；
- UI 界面选择文件夹并进行现场演示；
- 在原图上生成扣分项文字标注；
- 对模型关注区域生成 Grad-CAM 热力图；
- 支持普通 ResNet18 模型和 ResNet18 + CBAM 模型混合调用；
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
10. 作为农村人居环境现场评分的 AI 初评工具。

---

## 二、评分标准

系统按照“AI 积分制现场照片采样及评分细则”进行评分，总分为 60 分，分为四个主要部分：

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
A0 = 1
A1 = 1
A2 = 1
A3 = 0
A4 = 0
A5 = 0
A6 = 1
A7 = 1
A8 = 1
A9 = 0
```

对应到数据表中就是：

```text
label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9
1,1,1,0,0,0,1,1,1,0
```

其中：

```text
1 表示该项扣分
0 表示该项不扣分
```

---

### 3.2 为什么使用多标签分类

普通图像分类通常是一张图片只属于一个类别，例如：

```text
猫 / 狗 / 汽车 / 飞机
```

但本项目中，一张图片可能同时存在多个扣分项。

例如一张庭院照片可能同时存在：

```text
B1 庭院内交通用具杂乱无章
B5 庭院内地面垃圾乱丢现象严重
B10 庭院污水横流
```

因此模型不能只输出一个类别，而是需要输出多个标签的概率。

模型输出形式类似：

```text
B0: 0.41
B1: 0.78
B2: 0.52
B3: 0.10
B4: 0.42
B5: 0.71
...
```

系统再根据阈值判断是否扣分：

```text
概率 >= 阈值：判定该扣分项存在
概率 < 阈值：判定该扣分项不存在
```

例如：

```text
B1 概率 0.78，阈值 0.55，所以 B1 扣分
B5 概率 0.71，阈值 0.50，所以 B5 扣分
```

---

### 3.3 ResNet18 模型原理

本项目使用 ResNet18 作为基础图像识别模型。

ResNet18 可以理解为一个图像特征提取器，它会从图片中逐层提取视觉信息：

```text
原始图片
↓
边缘、颜色、纹理
↓
局部结构
↓
物体特征
↓
场景特征
↓
输出多个扣分项概率
```

ResNet18 的核心是残差连接。普通卷积网络层数变深后，容易出现梯度消失或训练退化问题。ResNet 通过短接结构将输入特征直接与卷积后的残差特征相加，使网络更容易训练。

残差块可以简化表示为：

```text
输入 x
↓
卷积层
↓
BN / ReLU
↓
卷积层
↓
F(x)
↓
F(x) + x
↓
输出
```

其中：

```text
x 是原始输入特征
F(x) 是卷积层学习到的残差特征
F(x) + x 是残差块最终输出
```

这种结构可以让模型不用重新学习完整特征，而是学习“需要修正的部分”，从而提升深层网络训练稳定性。

---

### 3.4 迁移学习

由于项目数据量较少，不适合从零开始训练一个深度神经网络，因此采用迁移学习方法。

迁移学习的核心思想是：

```text
使用已经在大规模图像数据上训练过的 ResNet18
保留它已有的基础视觉识别能力
只修改最后的分类层，让它适配本项目的扣分标签
```

原始 ResNet18 的输出是 1000 个 ImageNet 类别，本项目中需要改成：

```text
室内模型：输出 10 个标签
庭院模型：输出 12 个标签
厕所模型：输出 2 个标签
化粪池模型：输出 3 个标签
房前屋后模型：输出 5 个标签
```

---

### 3.5 ResNet18 + CBAM

庭院场景内容复杂，包含生产工具、交通工具、柴草、垃圾、鸡鸭、污水、棚架等多种目标，且很多扣分项只占图像局部区域。普通 ResNet18 容易受到复杂背景干扰，因此系统对庭院模型增加了 CBAM 注意力模块。

CBAM 由两部分组成：

```text
Channel Attention：通道注意力，判断哪些特征更重要
Spatial Attention：空间注意力，判断图像哪些区域更重要
```

庭院 CBAM 模型流程为：

```text
庭院图片
↓
ResNet18 提取基础图像特征
↓
CBAM 增强关键通道和关键区域
↓
全局平均池化
↓
多标签分类头
↓
输出 B0-B11 扣分项概率
```

在本项目中，CBAM 主要用于提升庭院场景中局部扣分项的识别能力，例如：

```text
地面垃圾
污水横流
柴草堆码
破败棚架
杂物堆放
```

注意：加了 CBAM 后，模型结构发生变化。普通 ResNet18 不能直接加载 CBAM 模型文件，预测代码需要自动判断模型是否包含 `cbam.xxx` 参数，并使用对应的模型结构加载。

---

### 3.6 Grad-CAM 热力图

当前系统新增了 Grad-CAM 热力图功能，用于解释模型在判断某个扣分项时主要关注了图像中的哪些区域。

Grad-CAM 的作用是：

```text
输入图片
↓
选择某个扣分项输出
↓
反向计算该扣分项对最后卷积特征图的贡献
↓
生成热力图
↓
叠加到原图上
```

热力图中红色越明显，表示模型在判断该扣分项时越关注该区域。

需要注意：

```text
Grad-CAM 热力图表示模型关注区域
不是人工标注框
不是目标检测框
不能等同于物体精确位置
```

如果后续需要精确框出垃圾、污水、柴草等物体，需要进一步训练 YOLO 等目标检测模型。

---

### 3.7 为什么不直接预测最终分数

本项目没有采用“输入图片后直接输出分数”的方式，而是采用：

```text
先识别扣分项
再根据规则计算分数
```

这种方式更适合现场应用，主要有三个优点：

第一，可解释性更强。

系统可以说明模型为什么扣分，例如：

```text
庭院扣分项：
B1 庭院内交通用具杂乱无章，扣 3 分
B5 庭院内地面垃圾乱丢现象严重，扣 5 分
庭院最终得分：30 - 8 = 22 分
```

第二，便于人工复核。

现场工作人员可以查看模型识别出的扣分项、文字标注图和 Grad-CAM 热力图，而不是只看到一个无法解释的分数。

第三，便于后续维护。

如果评分规则发生变化，只需要修改扣分权重或标签配置，不一定需要重新设计整个模型。

---

## 四、模型训练过程

### 4.1 训练数据来源

训练数据来自：

```text
data/all_labels.csv
```

每一行包含：

```text
house_id
scene
image_path
label_0 ~ label_11
```

对于某个场景模型，程序会先筛选对应场景的数据。

例如庭院模型只筛选：

```text
scene == 庭院
```

然后读取：

```text
庭院图片
庭院 label_0 ~ label_11
```

---

### 4.2 训练流程

训练流程如下：

```text
读取 all_labels.csv
↓
筛选指定场景的数据
↓
读取图片和人工标签
↓
图片预处理和数据增强
↓
输入 ResNet18 或 ResNet18 + CBAM 模型
↓
输出多个扣分项预测值
↓
使用 BCEWithLogitsLoss 计算损失
↓
反向传播更新模型参数
↓
保存训练好的 .pth 模型文件
```

其中使用的损失函数是：

```text
BCEWithLogitsLoss
```

这是因为每个标签本质上都是一个二分类任务：

```text
该扣分项是否存在？
是：1
否：0
```

多个标签合起来，就构成了多标签分类任务。

---

### 4.3 epoch、batch 和模型保存

训练命令中的 `epochs` 表示训练轮数。

例如：

```powershell
python -m src.train_courtyard_cbam_70 --csv data/all_labels.csv --epochs 40 --train_num 70
```

表示将训练集完整学习 40 轮。

如果训练集有 70 张图片，batch size 为 8，则每一轮大约包含：

```text
70 / 8 ≈ 9 个 batch
```

40 个 epoch 大约会进行：

```text
9 × 40 = 360 次参数更新
```

模型不是“训练 40 张图就结束”，而是把训练集反复学习 40 轮。每一轮中还会进行随机裁剪、翻转、旋转、亮度变化等数据增强，因此模型每次看到的图像都会略有不同。

训练完成后，程序会保存 `.pth` 文件，例如：

```text
models/courtyard_resnet18_cbam.pth
```

`.pth` 文件中主要包含：

```text
model_state_dict：模型参数
label_names：标签名称
label_cols：标签列
deducts：扣分权重
thresholds：预测阈值
train_house_ids：训练使用的农户编号
model_type：模型类型，例如 resnet18_cbam
```

---

## 五、现场应用流程

现场应用时，每户农户建议单独建立一个文件夹。

例如第 97 户：

```text
data/raw/97/
├── 97.csv
├── 室内_97.jpg
├── 庭院_97.jpg
├── 厕所_97.jpg
├── 化粪池_97.jpg
└── 房前屋后_97.jpg
```

其中：

| 文件              | 作用                                   |
| ----------------- | -------------------------------------- |
| `97.csv`          | 该户人工标注结果，用于计算人工实际得分 |
| `室内_97.jpg`     | 室内照片                               |
| `庭院_97.jpg`     | 庭院照片                               |
| `厕所_97.jpg`     | 厕所照片                               |
| `化粪池_97.jpg`   | 化粪池照片                             |
| `房前屋后_97.jpg` | 房前屋后照片                           |

现场演示流程：

```text
打开 UI 界面
↓
选择农户文件夹
↓
点击开始评分
↓
系统自动读取 CSV 计算人工实际得分
↓
系统自动读取图片进行 AI 预测
↓
输出每个场景实际得分、预测得分、分差
↓
输出总分实际值、预测值、总分分差
↓
生成带扣分项文字标注的图片
↓
生成 Grad-CAM 热力图
```

---

## 六、项目结构

```text
countryside_score/
│
├── src/                                      # 核心代码
│   ├── train_test_indoor.py                  # 室内模型训练与测试
│   ├── eval_indoor_current.py                # 室内模型重新测评
│   ├── predict_indoor.py                     # 单张室内图片预测
│   │
│   ├── train_outside_70.py                   # 房前屋后模型训练
│   ├── eval_outside_holdout.py               # 房前屋后保留集测评
│   ├── eval_outside_current.py               # 房前屋后当前模型整体测评
│   ├── predict_outside.py                    # 单张房前屋后图片预测
│   │
│   ├── train_courtyard_70.py                 # 普通庭院 ResNet18 模型训练
│   ├── train_courtyard_cbam_70.py            # 庭院 ResNet18 + CBAM 模型训练
│   ├── train_toilet_70.py                    # 厕所模型训练
│   ├── train_septic_70.py                    # 化粪池模型训练
│   │
│   ├── predict_septic.py                     # 单张化粪池图片预测
│   ├── predict_house_total.py                # 根据 house_id 进行整户预测
│   ├── predict_folder_total.py               # 输入文件夹进行整户评分
│   │
│   ├── update_thresholds.py                  # 修改模型阈值
│   ├── ui_folder_score.py                    # 简单 UI 界面
│   ├── ui_folder_score_annotated.py          # 带文字标注的 UI 界面
│   ├── ui_folder_score_annotated_cbam_fixed.py
│   │                                      # 支持庭院 CBAM 模型的 UI 界面
│   └── ui_folder_score_annotated_gradcam.py  # 支持文字标注和 Grad-CAM 热力图的 UI 界面
│
├── scripts/                                  # 数据整理脚本
│   └── build_dataset.py                      # 将每户 CSV 整理成 all_labels.csv
│
├── data/                                     # 数据目录，本地保存，不上传 Git
│   ├── raw/                                  # 原始农户照片与每户标注 CSV
│   └── all_labels.csv                        # 整理后的总标注表
│
├── models/                                   # 训练好的模型文件，本地保存，不上传 Git
│   ├── indoor_resnet18.pth                   # 室内模型
│   ├── courtyard_resnet18.pth                # 普通庭院模型
│   ├── courtyard_resnet18_cbam.pth           # 庭院 CBAM 模型
│   ├── toilet_resnet18.pth                   # 厕所模型
│   ├── septic_resnet18.pth                   # 化粪池模型
│   └── outside_resnet18.pth                  # 房前屋后模型
│
├── outputs/                                  # 测评结果输出
│
├── .gitignore                                # Git 忽略规则
├── requirements.txt                          # Python 依赖
└── README.md                                 # 项目说明文档
```

说明：

```text
data/raw/      存放原始图片和每户人工标注 CSV
models/        存放训练好的 .pth 模型
outputs/       存放批量测评结果
src/           存放训练、预测、UI 等核心代码
scripts/       存放数据整理脚本
```

---

## 七、环境安装

建议使用 Python 虚拟环境运行项目。

进入项目目录：

```powershell
cd D:\desktop\countryside_score
```

创建虚拟环境：

```powershell
python -m venv venv
```

激活虚拟环境：

```powershell
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

如果需要运行 UI 界面，`tkinter` 一般随 Python 自带，无需单独安装。

---

## 八、数据格式

### 8.1 总标注文件格式

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

字段说明：

| 字段                    | 含义                                             |
| ----------------------- | ------------------------------------------------ |
| `house_id`              | 农户编号                                         |
| `scene`                 | 场景名称，例如室内、庭院、厕所、化粪池、房前屋后 |
| `image_path`            | 图片路径                                         |
| `label_0` 到 `label_11` | 对应场景下的扣分标签                             |
| `1`                     | 表示该项扣分                                     |
| `0`                     | 表示该项不扣分                                   |

---

### 8.2 每户文件夹内 CSV 格式

用于 UI 演示时，每户文件夹中可以放一个单户 CSV，例如：

```text
data/raw/97/97.csv
```

格式如下：

```csv
序号,label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11
厕所,1,1,,,,,,,,,,
房前屋后,0,0,0,0,1,,,,,,,
室内,1,1,1,0,0,0,1,1,1,0,,
庭院,1,0,0,0,0,0,0,0,0,0,0,0
化粪池,0,0,1,,,,,,,,,
```

该 CSV 用于计算人工实际得分。

---

## 九、训练各场景模型

### 9.1 室内模型训练

室内场景对应 A0 到 A9，共 10 个标签。

```powershell
python -m src.train_test_indoor --csv data/all_labels.csv --epochs 30
```

训练完成后生成：

```text
models/indoor_resnet18.pth
indoor_test_result.csv
```

---

### 9.2 房前屋后模型训练

房前屋后场景对应 D0 到 D4，共 5 个标签。

使用前 70 组数据训练：

```powershell
python -m src.train_outside_70 --csv data/all_labels.csv --epochs 30 --train_num 70
```

使用前 60 组数据训练：

```powershell
python -m src.train_outside_70 --csv data/all_labels.csv --epochs 30 --train_num 60
```

训练完成后生成：

```text
models/outside_resnet18.pth
outside_holdout_rows.csv
```

---

### 9.3 普通庭院模型训练

庭院场景对应 B0 到 B11，共 12 个标签。

```powershell
python -m src.train_courtyard_70 --csv data/all_labels.csv --epochs 30 --train_num 70
```

训练完成后生成：

```text
models/courtyard_resnet18.pth
courtyard_holdout_rows.csv
```

---

### 9.4 庭院 ResNet18 + CBAM 模型训练

为了提升复杂庭院场景下局部扣分项识别效果，当前系统支持训练庭院 CBAM 模型。

```powershell
python -m src.train_courtyard_cbam_70 --csv data/all_labels.csv --epochs 40 --train_num 70
```

训练完成后生成：

```text
models/courtyard_resnet18_cbam.pth
courtyard_cbam_holdout_rows.csv
```

该模型主要用于庭院场景，在 UI 中会自动检测模型是否为 CBAM 结构，并用对应结构加载。

---

### 9.5 厕所模型训练

厕所模型只包含厕所部分，不包含化粪池。

对应标签：

```text
label_0：厕屋脏乱
label_1：厕屋功能配备不齐全
```

```powershell
python -m src.train_toilet_70 --csv data/all_labels.csv --epochs 30 --train_num 70
```

训练完成后生成：

```text
models/toilet_resnet18.pth
toilet_holdout_rows.csv
```

---

### 9.6 化粪池模型训练

化粪池模型对应 3 个标签：

```text
label_0：化粪池盖板挪开未关闭，取粪口未关闭
label_1：化粪池粪污溢流
label_2：厕所周围其他情况
```

```powershell
python -m src.train_septic_70 --csv data/all_labels.csv --epochs 30 --train_num 70
```

训练完成后生成：

```text
models/septic_resnet18.pth
septic_holdout_rows.csv
```

---

## 十、单张图片预测

### 10.1 单张室内图片预测

```powershell
python -m src.predict_indoor --image "data/raw/55/室内_55.jpg"
```

输出内容包括：

```text
A0 到 A9 每个标签的预测概率
每个标签是否扣分
室内总扣分
室内最终得分
```

---

### 10.2 单张房前屋后图片预测

```powershell
python -m src.predict_outside --image "data/raw/71/房前屋后_71.jpg"
```

输出内容包括：

```text
D0 到 D4 每个标签的预测概率
每个标签是否扣分
房前屋后总扣分
房前屋后最终得分
```

---

### 10.3 单张化粪池图片预测

```powershell
python -m src.predict_septic --image "data/raw/61/化粪池_61.jpg"
```

输出内容包括：

```text
化粪池各标签预测概率
每个标签是否扣分
化粪池部分扣分
```

注意：

```text
化粪池不单独计 10 分，需要与厕所扣分合并计算。
厕所及化粪池得分 = 10 - 厕所扣分 - 化粪池扣分
```

---

## 十一、输入文件夹进行整户评分

本项目支持直接输入一个农户文件夹进行整体测评。

```powershell
python -m src.predict_folder_total --folder data/raw/97
```

该命令会自动完成：

```text
读取 data/raw/97/97.csv
读取 data/raw/97/室内_97.jpg
读取 data/raw/97/庭院_97.jpg
读取 data/raw/97/厕所_97.jpg
读取 data/raw/97/化粪池_97.jpg
读取 data/raw/97/房前屋后_97.jpg
计算人工实际得分
计算 AI 模型预测得分
输出每个场景的实际分、预测分和分差
输出总分对比
```

输出示例：

```text
========== 评分结果 ==========
      项目  满分  实际扣分  预测扣分  实际得分  预测得分  分差
      室内  10      6      4      4      6    2
      庭院  30      3      8     27     22   -5
厕所及化粪池 10      6      6      4      4    0
  房前屋后 10      1      1      9      9    0
      总分  60     16     19     44     41   -3
```

其中：

```text
分差 = 预测得分 - 实际得分
```

含义：

```text
分差 > 0：AI 给分偏高
分差 < 0：AI 给分偏低
分差 = 0：AI 与人工一致
```

---

## 十二、UI 界面运行

当前推荐使用支持 CBAM 和 Grad-CAM 的 UI：

```powershell
python -m src.ui_folder_score_annotated_gradcam
```

运行后会弹出桌面窗口。

操作步骤：

```text
点击“选择农户文件夹”
选择 data/raw/97 这类文件夹
点击“开始评分”
查看评分结果
查看 generated annotated 文字标注图
查看 generated gradcam 热力图
```

UI 界面会显示：

| 项目         | 满分 | 实际扣分 | 预测扣分 | 实际得分 | 预测得分 | 分差 |
| ------------ | ---: | -------: | -------: | -------: | -------: | ---: |
| 室内         |   10 |        6 |        4 |        4 |        6 |    2 |
| 庭院         |   30 |        3 |        8 |       27 |       22 |   -5 |
| 厕所及化粪池 |   10 |        6 |        6 |        4 |        4 |    0 |
| 房前屋后     |   10 |        1 |        1 |        9 |        9 |    0 |
| 总分         |   60 |       16 |       19 |       44 |       41 |   -3 |

UI 运行后会在农户文件夹下生成：

```text
data/raw/97/annotated/
data/raw/97/gradcam/
```

---

## 十三、图片扣分项文字标注

UI 版本支持在图片上自动写出模型预测出的扣分项。

例如庭院图片被模型判断存在：

```text
B1 庭院内交通用具杂乱无章
B5 庭院内地面垃圾乱丢现象严重
```

系统会在图片上生成文字标注：

```text
场景：庭院
模型扣分：8
扣分项：
B1_庭院内交通用具杂乱无章 | 概率 0.778 | -3 分
B5_庭院内地面垃圾乱丢现象严重 | 概率 0.716 | -5 分
```

生成结果会保存在：

```text
data/raw/97/annotated/
```

例如：

```text
data/raw/97/annotated/室内_97_标注.jpg
data/raw/97/annotated/庭院_97_标注.jpg
data/raw/97/annotated/厕所_97_标注.jpg
data/raw/97/annotated/化粪池_97_标注.jpg
data/raw/97/annotated/房前屋后_97_标注.jpg
```

---

## 十四、Grad-CAM 热力图

系统新增 Grad-CAM 热力图功能，用于展示模型在判断某个扣分项时关注的图像区域。

运行 UI 后会生成：

```text
data/raw/97/gradcam/
```

例如：

```text
data/raw/97/gradcam/庭院_97_B5_庭院内地面垃圾乱丢现象严重_热力图.jpg
data/raw/97/gradcam/室内_97_A6_室内地面垃圾乱丢现象严重_热力图.jpg
```

热力图含义：

```text
红色越明显，表示模型在判断该扣分项时越关注该区域
```

需要注意：

```text
Grad-CAM 是模型关注区域可视化
不是目标检测框
不能完全等同于问题物体的精确位置
```

如果后续需要框出垃圾、污水、柴草等具体目标，需要训练 YOLO 等目标检测模型。

---

## 十五、模型测评

### 15.1 室内模型测评

```powershell
python -m src.eval_indoor_current --csv data/all_labels.csv --model models/indoor_resnet18.pth
```

输出指标包括：

```text
标签平均准确率
整张图片标签完全一致率
Macro F1
Macro Precision
Macro Recall
平均得分误差
各标签 F1
```

---

### 15.2 房前屋后模型测评

使用保留数据测评：

```powershell
python -m src.eval_outside_holdout --holdout outside_holdout_rows.csv --model models/outside_resnet18.pth
```

对当前全部房前屋后数据重新测评：

```powershell
python -m src.eval_outside_current --csv data/all_labels.csv --model models/outside_resnet18.pth
```

测评结果会保存为 CSV 文件。

---

### 15.3 庭院模型测评建议

当前庭院模型支持普通 ResNet18 和 ResNet18 + CBAM 两种版本。

建议对比：

```text
courtyard_resnet18.pth
courtyard_resnet18_cbam.pth
```

重点比较：

```text
庭院实际得分
庭院预测得分
庭院分差
B0-B11 各标签预测情况
误扣和漏扣样本
```

---

## 十六、阈值调整

模型输出的是概率，系统需要根据阈值判断是否扣分。

例如：

```text
概率 >= 阈值：扣分
概率 < 阈值：不扣分
```

如果模型经常扣分太多，说明预测得分低于人工得分，可以适当调高阈值。

如果模型经常漏扣，说明预测得分高于人工得分，可以适当调低阈值。

调整方向：

```text
预测得分 < 实际得分：模型扣分太多 → 阈值调高
预测得分 > 实际得分：模型扣分太少 → 阈值调低
```

例如庭院中 B5 是 5 分大项，如果误扣会导致分差很大，因此可以适当提高 B5 的阈值，减少误扣。

如果需要直接修改模型文件中的阈值，可以使用：

```powershell
python -m src.update_thresholds --model models/courtyard_resnet18_cbam.pth --thresholds "0.40,0.75,0.65,0.50,0.55,0.75,0.60,0.60,0.60,0.65,0.60,0.75"
```

---

## 十七、现场应用价值

本系统在现场应用中具有以下价值：

### 17.1 提高评分效率

传统人工评分需要工作人员逐张照片查看，并根据评分表手动判断扣分项。系统可以自动读取照片并输出初步评分结果，减少重复性工作。

### 17.2 保持评分规则统一

人工评分容易受经验、主观感受和现场环境影响。AI 系统按照统一阈值和统一扣分规则进行计算，有助于提高评分一致性。

### 17.3 支持人工复核

系统不会完全替代人工，而是作为 AI 初评工具。工作人员可以查看模型预测结果、分差、文字标注图和 Grad-CAM 热力图，再决定是否修正。

### 17.4 便于演示和归档

系统可以输出 CSV 结果、带标注图片和热力图，方便用于现场演示、项目汇报和结果归档。

---

## 十八、Git 管理说明

本项目使用 Git 管理代码，但不建议上传大文件。

不上传 Git 的内容包括：

```text
venv/
data/raw/
models/
outputs/
*.pth
*.pt
*.onnx
*.jpg
*.jpeg
*.png
*.bmp
*.webp
*_result.csv
*_eval*.csv
annotated/
gradcam/
```

建议上传 Git 的内容包括：

```text
src/
scripts/
README.md
requirements.txt
.gitignore
config.py
```

推荐提交命令：

```powershell
git add .gitignore README.md requirements.txt src/ scripts/ config.py
git commit -m "更新农户环境AI评分系统：CBAM与Grad-CAM可解释模块"
```

---

## 十九、注意事项

1. 训练和预测建议始终在项目根目录运行，不要进入 `src/` 后再运行脚本。
2. 推荐使用 `python -m src.xxx` 的方式运行代码。
3. 图片路径应相对于项目根目录，例如 `data/raw/71/房前屋后_71.jpg`。
4. 每户文件夹中的 CSV 应包含 `序号` 或 `scene` 列，并包含 `label_0` 到对应标签列。
5. 数据量较少时，模型结果只能作为 AI 初评，建议保留人工复核。
6. 如果重新训练模型，默认会覆盖 `models/` 中同名模型文件。
7. 若更换电脑，需要重新创建虚拟环境并安装依赖。
8. 当前图片标注是扣分项文字标注，不是目标检测框。
9. Grad-CAM 热力图是模型关注区域，不是精确目标框。
10. 若需要检测具体物体位置，需要后续引入 YOLO 等目标检测模型。
11. 庭院、厕所、化粪池等场景中部分标签样本较少时，模型效果可能不稳定，需要继续补充数据或调整阈值。
12. CBAM 模型和普通 ResNet18 模型结构不同，加载时必须使用匹配的网络结构。
13. 当前 UI 代码已经支持自动判断普通 ResNet18 和 CBAM 模型，但模型路径必须配置正确。

---

## 二十、项目总结

本项目将农村人居环境积分制评分规则转化为一个可运行的 AI 图像评分系统。

系统通过多标签分类模型识别每张现场照片中的扣分项，再按照人工评分细则计算每个场景得分和总分。与直接预测分数相比，本系统具有更好的可解释性和可复核性。

当前系统已经实现：

```text
现场照片输入
↓
扣分项自动识别
↓
场景得分计算
↓
人工分与预测分对比
↓
分差分析
↓
带扣分项文字标注图片输出
↓
Grad-CAM 热力图输出
↓
UI 界面现场演示
```

本系统适合作为农村人居环境现场评分的 AI 初评工具，也可以作为后续目标检测、自动标注和智能巡检系统的基础版本。