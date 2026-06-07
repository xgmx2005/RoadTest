# RoadTest

基于 **CULane 预训练 UFLD（Ultra-Fast-Lane-Detection）** 与 **CLRNet** 的道路车道线检测项目。

功能：
- 输入司机视角路面图片
- 检测车道线
- 判断每条车道线颜色（黄色 / 白色）
- 输出带标注的可视化结果图
- 支持批量处理并统计平均单张耗时

## 输入数据（已随仓库附带）

仓库已包含两组输入测试图，拉下来即可直接运行，无需另外准备图片：
- `test-difficult/`：37 张难例测试图（老师测试集）
- `道路example/`：50 张示例测试图

> CLRNet 改进模型的 ONNX（`models/clrnet_culane_r18.onnx`）也已附带，配合上面的输入图可**完全离线**跑通。仅 UFLD 基线的权重 `culane_18.pth`（约 170MB）因体积过大未入库，需按下方说明用 `gdown` 下载。

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

### UFLD 基线模型

```bash
python run_batch.py --input test-difficult --output out_difficult --weights culane_18.pth
```

```bash
python run_batch.py --input 道路example --output out_example --weights culane_18.pth
```

### CLRNet 改进模型

仓库已包含 CPU 可运行的 `models/clrnet_culane_r18.onnx`：

```bash
python run_batch_clrnet.py --input test-difficult --output out_difficult_clrnet
```

输出内容：
- `*_vis.jpg`：可视化结果图
- `results.json`：逐图检测结果与时间统计

仓库中已附带实际运行后的结果：
- `results/test-difficult/`：UFLD + 改进颜色判别结果
- `results/example/`：UFLD + 改进颜色判别结果
- `results/test-difficult-clrnet/`：CLRNet + 改进颜色判别 + 轻量误检过滤结果

## 文件说明

- `ufld_model.py`：UFLD 模型定义
- `lane_detector.py`：UFLD 检测、颜色分类、可视化
- `clr_detector.py`：CLRNet ONNX 检测，复用改进后的黄/白颜色分类，并含轻量边界误检过滤
- `run_batch.py`：UFLD 批量推理与时间统计
- `run_batch_clrnet.py`：CLRNet 批量推理与时间统计
- `models/clrnet_culane_r18.onnx`：CLRNet CULane ResNet18 ONNX 模型
- `results/`：本次测试生成的结果图与 `results.json`
- `docs/report_期末实验说明.md`：实验说明与结果写法
- `docs/final_report_期末实验正文.md`：可直接提交的实验报告正文
