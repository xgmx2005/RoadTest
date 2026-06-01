# 道路车道线检测期末实验说明（Anaconda 版本）

## 1. 整体方案
本方案满足课程实验的核心要求：

1. 输入司机视角道路图像
2. 检测图中的车道线
3. 判断每条车道线颜色（黄色 / 白色）
4. 输出可视化结果图，并统计批量处理效率

采用方法：
- **车道线检测模型**：UFLD（Ultra-Fast-Lane-Detection）
- **预训练权重**：CULane ResNet18
- **颜色分类**：在检测出的车道线点附近采样亮像素，结合 BGR 比值和 HSV 判断黄/白

## 2. Anaconda 环境
```bash
conda env create -f environment.yml
conda activate lane_det
```

如果是 CPU：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. 下载模型权重
```bash
gdown 1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq -O culane_18.pth
```

## 4. 运行方式
### 4.1 运行 test-difficult
```bash
python run_batch.py --input test-difficult --output out_difficult --weights culane_18.pth
```

### 4.2 运行 道路example
```bash
python run_batch.py --input 道路example --output out_example --weights culane_18.pth
```

程序输出：
- 每张图的 `*_vis.jpg`
- 一个 `results.json`

## 5. 本次实际跑出的结果
### test-difficult（37 张）
- 检测到车道线总数：91
- 黄色：14
- 白色：77
- 总耗时：1.028 s
- 平均单张耗时：27.8 ms
- 速度：35.99 FPS

### 道路example（50 张）
- 检测到车道线总数：138
- 黄色：6
- 白色：132
- 总耗时：1.302 s
- 平均单张耗时：26.0 ms
- 速度：38.39 FPS

## 6. 人工评估（test-difficult）
按“司机当前方向主要道路”进行保守人工判断：

### 6.1 车道线检测
- 有效预测车道线数：89
- TP：83
- FP：6
- FN：2

计算：
- Precision = 83 / 89 = 93.3%
- Recall = 83 / 85 = 97.6%

### 6.2 黄色车道线分类
- 黄色线 TP：6
- 黄色线 FP：3
- 黄色线 FN：4

计算：
- Yellow Precision = 6 / 9 = 66.7%
- Yellow Recall = 6 / 10 = 60.0%

## 7. 结果分析
- 几何检测效果较稳定，说明 CULane 预训练模型对主车道边界提取较有效。
- 黄色/白色分类受逆光、偏暖色温、车头反光、路面旧化影响较大，因此颜色分类精度低于几何检测。
- 该方案优点是实现简单、推理速度快、满足课程实验对功能和效率的要求。

## 8. 可继续优化的方向
1. 先做白平衡或颜色归一化，再做黄/白分类
2. 对车头区域、护栏边缘、文字区域做 ROI 过滤
3. 只对车道线中心附近像素做更精细采样
4. 如果时间允许，可进一步做人工标注和自动化评估
