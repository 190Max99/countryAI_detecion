# predict_outside.py 推理流程图

```mermaid
flowchart TD
    A["🚀 启动 main()"] --> B["解析命令行参数<br/>--image (必填) / --model (默认 outside_resnet18.pth)"]
    B --> C{"图片路径存在?"}
    C -- 否 --> D["❌ FileNotFoundError"]
    C -- 是 --> E{"模型文件存在?"}
    E -- 否 --> F["❌ FileNotFoundError"]
    E -- 是 --> G["设置设备 GPU / CPU"]

    G --> H["加载 checkpoint<br/>load_checkpoint()<br/>weights_only=False 兼容 PyTorch 2.6+"]
    H --> I["从 checkpoint 提取元信息<br/>label_names / deducts / thresholds"]
    I --> J["构建模型 build_model()<br/>ResNet18(weights=None) → fc=Linear(512→5)"]
    J --> K["加载模型权重<br/>model.load_state_dict()"]
    K --> L["model.eval() 推理模式"]

    L --> M["加载图片<br/>Image.open() → RGB"]
    M --> N["图像预处理 get_transform()<br/>Resize(224)→ToTensor→Normalize<br/>unsqueeze(0) 增加 batch 维度"]
    N --> O["送入 GPU / CPU"]

    O --> P["🧠 模型推理<br/>with torch.no_grad():"]
    P --> Q["logits = model(image)"]
    Q --> R["probs = sigmoid(logits)<br/>输出 5 个标签概率 [0~1]"]

    R --> S["与阈值比较<br/>pred_labels = (probs ≥ thresholds)"]
    S --> T["遍历 5 个标签计算扣分"]

    T --> U{"prob ≥ threshold?"}
    U -- 是 --> V["total_deduct += deducts[i]<br/>标记为扣分"]
    U -- 否 --> W["不扣分"]
    V --> X{"还有标签?"}
    W --> X
    X -- 是 --> T
    X -- 否 --> Y["final_score = max(0, 10 - total_deduct)"]

    Y --> Z["✅ 输出结果<br/>每个标签: 概率 / 阈值 / 是否扣分<br/>总扣分 + 最终得分 (满分10)"]

    style A fill:#4CAF50,color:#fff
    style Z fill:#4CAF50,color:#fff
    style D fill:#f44336,color:#fff
    style F fill:#f44336,color:#fff
    style P fill:#9C27B0,color:#fff
    style Y fill:#2196F3,color:#fff
```

## 阶段说明

| 阶段           | 说明                                                                                |
| -------------- | ----------------------------------------------------------------------------------- |
| **参数校验**   | 检查 `--image` 图片和 `--model` 模型文件是否存在                                    |
| **模型加载**   | 读取 checkpoint → 提取标签名/扣分值/阈值 → 构建 ResNet18 → 加载权重 → `eval()` 模式 |
| **图像预处理** | Resize(224)→ToTensor→ImageNet 归一化（**无数据增强**，与训练不同）                  |
| **推理计算**   | `torch.no_grad()` 下前向传播 → sigmoid 转概率 → 与阈值比较判定是否扣分              |
| **结果输出**   | 逐标签打印概率/阈值/扣分情况 → 汇总总扣分 → 输出最终得分                            |

## 与训练脚本的关键区别

| 对比项       | `train_outside_70.py`                  | `predict_outside.py`    |
| ------------ | -------------------------------------- | ----------------------- |
| **用途**     | 训练模型                               | 单张图片推理            |
| **数据增强** | ✅ RandomCrop/Flip/Rotation/ColorJitter | ❌ 仅 Resize + Normalize |
| **模型模式** | `model.train()`                        | `model.eval()`          |
| **梯度计算** | 需要反向传播                           | `torch.no_grad()` 禁用  |
| **输出**     | loss 值                                | 每个标签概率 + 得分     |

## 5 个分类标签

| 标签    | 名称                            | 扣分 | 阈值 |
| ------- | ------------------------------- | ---- | ---- |
| label_0 | D0_房屋旁柴草堆码乱堆不整齐     | 3    | 0.55 |
| label_1 | D1_房屋周身存在污水横流现象     | 2    | 0.50 |
| label_2 | D2_房屋周身瓜果棚架破败不堪     | 2    | 0.50 |
| label_3 | D3_房屋周身鸡鸭棚圈破败不堪脏臭 | 2    | 0.50 |
| label_4 | D4_房屋周身其他情况             | 1    | 0.65 |

> **评分公式**: 最终得分 = max(0, 10 - 总扣分)
