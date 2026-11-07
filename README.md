# SatRain precipitation retrieval baselines

SatRain GMI 降水反演的可复现代码与韩国测试集基线结果。当前仓库包含 SatRain 官方 CNN/U-Net 示例的模型代码、本地运行入口、GPROF-NN V8 预训练模型的 SatRain 适配脚本，以及两者在共同像素上的比较。仓库还收录一项独立的 ERA5-Land 温度降尺度实验的 diffusion（EDM）和 flow matching **代码与评估结果**。

## 当前结果

CNN 与 GPROF-NN 在 SatRain 韩国测试集的 **600 个场景、17,392,665 个共同有效像素**上评估，预测覆盖了 99.9973% 的合格雷达像素。各模型采用相同的雷达参考、空间网格、质量掩码和降水率阈值。以下为全部场景累计像素得到的指标，单位为 mm/h；完整数字见 [results/korea_baselines](results/korea_baselines/)。

| 方法 | MAE ↓ | RMSE ↓ | Bias |
| --- | ---: | ---: | ---: |
| CNN，原始输出 | 0.1909 | 1.3960 | -0.0398 |
| CNN，负值截为零 | 0.1719 | 1.3955 | -0.0209 |
| GPROF-NN V8，ancillary NONE | **0.1286** | **1.1774** | -0.0268 |
| 始终预测零 | 0.1784 | 1.6917 | -0.1784 |

| 降水率阈值 | 方法 | CSI ↑ | POD ↑ | FAR ↓ |
| --- | --- | ---: | ---: | ---: |
| 0.1 mm/h | CNN | 0.3074 | 0.7803 | 0.6635 |
| 0.1 mm/h | GPROF-NN | **0.5420** | **0.8020** | **0.3743** |
| 10 mm/h | CNN | 0.0221 | 0.0223 | **0.3094** |
| 10 mm/h | GPROF-NN | **0.2673** | **0.3046** | 0.3136 |

这个比较使用同一测试场景和同一评价像素。CNN 在 SatRain 的 `xs` 训练子集上训练；GPROF-NN 使用外部预训练权重，因此**训练数据和训练预算不同**。CNN 使用 gridded GMI，GPROF-NN 使用同次过境的 on-swath GMI；后者采用 `ancillary NONE`，不是完整辅助数据配置。这里的 CSI/POD/FAR 均由降水率阈值计算，未使用 CNN 的概率头。

## ERA5-Land 温度降尺度结果

独立实验的 EDM diffusion 和 conditional flow matching 在相同的 200 个测试样本上评估，结果见 [`results/era5_downscaling/`](results/era5_downscaling/)，训练与评估代码见 [`scripts/`](scripts/)，运行步骤见 [`ERA5_LAND_DOWNSCALING.md`](ERA5_LAND_DOWNSCALING.md)。该实验的目标变量为 2 米气温，单位 K；不属于 SatRain 降水反演。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| [`satrain_tools/`](satrain_tools/) | SatRain 下载后检查、统计、可视化工具 |
| [`cnn_official/`](cnn_official/) | 官方 CNN 模型、训练入口、韩国共同像素比较脚本 |
| [`cnn_official/upstream/`](cnn_official/upstream/) | 固定版本的 SatRain 原始 notebook 及其 MIT 许可证 |
| [`gprof_korea/`](gprof_korea/) | 韩国场景准备、GPROF-NN 推理与评估 |
| [`scripts/`](scripts/) | ERA5-Land 数据准备、EDM 和 flow matching 的模型、训练及评估代码 |
| [`vendor/edm/`](vendor/edm/) | 运行上述模型所需的 EDM 网络代码、项目扩展及上游许可证 |
| [`ERA5_LAND_DOWNSCALING.md`](ERA5_LAND_DOWNSCALING.md) | ERA5-Land 实验的运行步骤 |
| [`requirements-downscaling.txt`](requirements-downscaling.txt) | ERA5-Land 实验的 Python 依赖 |
| [`results/korea_baselines/`](results/korea_baselines/) | 小型汇总结果及实验说明 |
| [`results/era5_downscaling/`](results/era5_downscaling/) | ERA5-Land 温度降尺度的 EDM 与 flow matching 评估表及图片 |

原始数据、模型权重、逐场景预测、虚拟环境和未选入 `results/` 的历史实验输出均保留在本机，不随仓库发布。ERA5-Land 的训练权重也不在仓库中，因此重新评估前需先训练模型或提供自己的权重。本机版 `convolutional_neural_network_local.ipynb` 含有绝对路径与保存的输出，故不纳入 Git；可参考公开的原始 notebook 和 `cnn_official` 脚本。原始 notebook 与提取的模型代码来源、版本和 SHA256 见 [`cnn_official/source_info.json`](cnn_official/source_info.json)。

## 从头复现

在仓库根目录执行。以下是 Windows PowerShell 示例；Linux/macOS 请使用相应的 Python 可执行文件和路径。

1. 安装 SatRain 1.2.1 和 CNN 依赖。准备 `data/satrain_store` 目录后，下载 `gmi / xs / gridded` 的 training、validation 输入及自动配套的 target。`xs` 训练集有 539 个场景，validation 有 82 个场景。下载详情见 [`cnn_official/README.md`](cnn_official/README.md)。
2. 用 `python -m cnn_official.run --check` 检查读取流程，再用 `python -m cnn_official.run --epochs 20 --batch-size 2 --accelerator auto` 训练 CNN。这个本地运行配置的 batch size 为 2；SatRain 原始 notebook 的示例配置为 32。训练产生的权重位于 `cnn_official/runs/`，不在 Git 中。
3. 在独立的 GPROF-NN V8 环境准备模型 `gprof_nn_3d_gmi.pt`，按 [`gprof_korea/README.md`](gprof_korea/README.md) 运行 `prepare --all`、`infer --all`、`evaluate --all`，下载并评估韩国测试集。GPROF-NN 权重来自其预训练模型，本项目不重新训练它。
4. 两边结果齐备后，运行 `python -m cnn_official.compare_korea --checkpoint <你的CNN权重路径>`。脚本需要完整的 600 场景 GPROF-NN 评估文件；输出写到 `experiments/cnn_gprof_korea_comparison/`。详细条件见 [`cnn_official/KOREA_COMPARISON.md`](cnn_official/KOREA_COMPARISON.md)。

首次运行前需在各自环境安装 PyTorch 及相应依赖；SatRain 与 GPROF-NN 使用独立环境。代码运行和数据下载由使用者执行。

SatRain 数据准备示例（PowerShell）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\cnn_official\requirements.txt
New-Item -ItemType Directory -Force .\data\satrain_store | Out-Null
.\.venv\Scripts\satrain.exe download --data_path .\data\satrain_store --base_sensors gmi --subset xs --splits training --geometries gridded --inputs gmi
.\.venv\Scripts\satrain.exe download --data_path .\data\satrain_store --base_sensors gmi --subset xs --splits validation --geometries gridded --inputs gmi
.\.venv\Scripts\python.exe -m satrain_tools.check_satrain --split validation
```

这些命令只准备 GMI CNN 所需的输入及对应 target。GPROF-NN 环境和韩国测试集的步骤见 [`gprof_korea/README.md`](gprof_korea/README.md)。

## 来源和使用范围

SatRain 原始 notebook 与提取的模型定义保留其上游 MIT 许可证，见 [`cnn_official/upstream/LICENSE.txt`](cnn_official/upstream/LICENSE.txt)。`vendor/edm/` 中的 NVIDIA 上游代码来自 [NVlabs/edm](https://github.com/NVlabs/edm)，采用 CC BY-NC-SA 4.0，详见 [`vendor/edm/LICENSE.txt`](vendor/edm/LICENSE.txt) 与其 [来源说明](vendor/edm/README.md)。其余项目脚本尚未指定复用许可证；发布者可在确定条款后另加根目录 `LICENSE`。
