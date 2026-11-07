# SatRain 官方 CNN 示例：本地入口

当前采用官方基线，入口是本目录。`cnn_official/run.py` 用于本地训练，`compare_korea.py` 用于共同像素评价。

## 本机个人 Notebook（不随仓库发布）

如果在本机保留了 `convolutional_neural_network_local.ipynb`，可在 VS Code 右上角选择本项目
`.venv/Scripts/python.exe` 为内核，按顺序逐格运行。这个个人版本包含官方输入配置、三个原样复制的模型定义单元格、
训练/加载权重、验证集预览、韩国场景 516 地图及可选的全部 600 场景评估。克隆公开仓库的读者可从
`upstream/convolutional_neural_network.ipynb` 和下文的 `run.py` 开始。

- 默认 `MODE="load"`，加载 `runs/colab_gmi_20261105_221329/checkpoints/best-epoch=18.ckpt`。这是较早通过 cnn_official.run 完成的 Colab 实验（batch size 2、FP32），不是后来的官方 Notebook batch size 32 实验。
- 如果需要重新训练，将 `MODE` 改为 `"train"`，保留官方 batch size 32、20 epochs。Windows workers=0，CPU FP32，GPU 16-mixed。
- CPU 内存不足时可以自行缩小 batch size，但需要记录为与官方配置的差异。
- 输出写入 `runs/notebook_<mode>_<UTC时间>/`。训练后默认用最终轮权重绘图，最佳权重另存。
- 韩国地图需要 Cartopy；当前 `.venv` 若未安装，请自行运行 `.\.venv\Scripts\python.exe -m pip install cartopy`。
- 韩国 Evaluator 单元格会下载缺少的测试文件；完整 600 场景推理需要在最后一格设置 `RUN_FULL_KOREA=True`。
- 这个本机版 Notebook 含有本机绝对路径和运行输出，Git 仓库中不发布；上游原始 Notebook 位于 `upstream/`。

## 官方源码

- 原始 Notebook：`upstream/convolutional_neural_network.ipynb`，按原文件保存。
- 模型代码：`official_model.py`，原样提取 Notebook 的第 12、13、15 号 cell（从 0 编号）。
- 类 `ResNetBlock`、`UNet`、`IPWGUNet` 的代码没有改写。
- 官方 MIT 许可证：`upstream/LICENSE.txt`。
- 来源与 SHA256：`source_info.json`。
- 固定版本：SatRain v1.2.1，Git commit `1b0f40676724e1bb00cc9b32f621e9e0fcc1522e`。

官方说明：
https://satrain.readthedocs.io/en/latest/notebooks/pytorch/convolutional_neural_network.html

## 使用的实验配置

遵循官方示例：

- 只使用 GMI 的 13 个观测通道，`include_angles=False`。
- `GMI(normalize="minmax", nan=-1.5)`，使用 SatRain 包内的统计量。
- 官方 `SatRainSpatial` 和 `TargetConfig(min_rqi=0.5)`。
- U-Net 特征宽度 `[32, 64, 128, 256, 512]`，BatchNorm 和 ReLU。
- 三个输出：降水率、降水发生的 logit、强降水发生的 logit。后两者应用 sigmoid 后才是概率。
- 官方损失：有效标签像素上的降水 MSE + 降水检测 BCE + 强降水检测 BCE。
- 官方 Adam（学习率 0.001）和余弦退火调度器。

这里没有增加 Softplus 或自定义输入掩码通道。官方降水输出层没有非负约束。
缺失输入用官方设定的 -1.5 表示，质量不合格的降水标签由官方加载器置为 NaN。
此前计算的 z-score 统计量仍保存在数据目录，但本配置使用官方 minmax 预处理。
GEO、ancillary 文件仍可用于后续多模态扩展；本轮 GMI 官方基线不读取它们。

## 本地适配

`run.py` 只负责实例化官方类并传入运行参数，不自写网络、损失或优化器。
它相对网页训练配置的改动是：

- 数据路径指向已有 `data/satrain_store`，`download=False`。
- 默认 batch size 为 2、num_workers 为 0，适合当前 Windows 环境；网页为 32 和 8。
- CPU 使用 `32-true`；CUDA 可用时使用 `16-mixed`。网页使用 GPU 半精度。
- 添加本地运行配置、CSV 日志和 ModelCheckpoint。
- `--check` 关闭训练数据增强，只检查真实样本与官方前向/损失；普通训练沿用官方数据增强。
- `--smoke` 使用 Lightning 的 `fast_dev_run`，运行一个训练批次和一个验证批次，不保存训练模型。

这属于官方模型在本地小数据上的运行，不声称复现网页中独立测试区域的数值结果。
模型类来源已按文本核对；训练和韩国测试已由用户实际完成，结果概览见根目录 README。

## 按顺序运行

在项目根目录的 PowerShell 中执行：

```powershell
cd <克隆后的仓库目录>
.\.venv\Scripts\python.exe -m pip install -r .\cnn_official\requirements.txt
.\.venv\Scripts\python.exe -m cnn_official.run --check
```

第一条安装命令补充 Lightning，SatRain 固定为已使用的 1.2.1。已有的 torch 和 torchvision 保持由 pip 的依赖解析处理。

检查通过后运行一批训练和验证：

```powershell
.\.venv\Scripts\python.exe -m cnn_official.run --smoke --accelerator cpu --batch-size 2
```

再运行 xs 数据训练：

```powershell
.\.venv\Scripts\python.exe -m cnn_official.run --epochs 20 --batch-size 2 --accelerator auto
```

官方模型约有 830 万参数，CPU 上完整训练可能较慢。`--output-dir` 可指定新的运行目录，默认写入 `cnn_official/runs` 下的时间戳目录。
普通训练按官方 `val_loss`（三个任务损失之和）选择最佳检查点，而非按单独的降水 MSE。

原始 Notebook 也可以在 VS Code 中阅读。其后半段含 `Evaluator(download=True)`，会下载独立测试集；
当前本地入口只使用已下载的训练和验证集。
