# RoadTest

基于 **CULane 预训练 UFLD（Ultra-Fast-Lane-Detection）** 的道路车道线检测项目。

功能：
- 输入司机视角路面图片
- 检测车道线
- 判断每条车道线颜色（黄色 / 白色）
- 输出带标注的可视化结果图
- 支持批量处理并统计平均单张耗时

## 环境（Anaconda）

```bash
conda env create -f environment.yml
conda activate lane_det
```

如果你想手动安装：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 下载模型权重

下载 CULane ResNet18 预训练权重：

```bash
gdown 1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq -O culane_18.pth
```

## 运行

### test-difficult
```bash
python run_batch.py --input test-difficult --output out_difficult --weights culane_18.pth
```

### 道路example
```bash
python run_batch.py --input 道路example --output out_example --weights culane_18.pth
```

输出内容：
- `*_vis.jpg`：可视化结果图
- `results.json`：逐图检测结果与时间统计

## 文件说明

- `ufld_model.py`：UFLD 模型定义
- `lane_detector.py`：车道线检测、颜色分类、可视化
- `run_batch.py`：批量推理与时间统计
- `docs/report_期末实验说明.md`：实验说明与结果写法
