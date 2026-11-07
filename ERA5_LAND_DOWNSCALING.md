# ERA5-Land 温度降尺度实验

这是独立于 SatRain 降水反演的实验。输入是 ERA5-Land 低分辨率 2 米气温，目标是同一时刻的高分辨率气温。代码包括条件 EDM diffusion 和 conditional flow matching；两者都使用 [`vendor/edm/`](vendor/edm/) 中的 EDM U-Net 代码。已保存的评估结果见 [`results/era5_downscaling/`](results/era5_downscaling/)；原始数据和模型权重不随仓库发布。

## 数据与环境

默认区域为北纬 29.5°–55°、西经 120°–94.5°，时间为 2025-01-01 00:00 至 2025-04-10 23:00 UTC，共 2,400 个逐小时样本。先配置 CDS API 账户，再在仓库根目录运行以下 PowerShell 命令；数据请求需遵守 CDS 的使用条件。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-downscaling.txt
.\.venv\Scripts\python.exe .\scripts\download_era5.py --dry-run
.\.venv\Scripts\python.exe .\scripts\download_era5.py
.\.venv\Scripts\python.exe .\scripts\prepare_dataset.py
```

下载文件保存在 `data/raw/`；预处理文件保存在 `data/era5_downscaling/`。预处理脚本要求连续且无重复的 2,400 个小时以及一致的 256×256 网格。每个高分辨率场按 8×8 区块平均得到 32×32 低分辨率场。划分按时间顺序进行：训练 2,000、验证 200、测试 200；归一化统计量只用训练集计算。`data/` 被 Git 忽略。

## 训练

下列命令分别训练两个模型并保存验证损失最优的权重。可用 `--device cpu` 或 `--device cuda` 指定设备。默认训练 20 轮、batch size 为 1；完整训练可能耗时较长。

```powershell
.\.venv\Scripts\python.exe .\scripts\train_edm.py --epochs 20 --batch-size 1
.\.venv\Scripts\python.exe .\scripts\train_flow.py --epochs 20 --batch-size 1
```

默认权重路径是 `experiments/edm_best.pt` 和 `experiments/flow_best.pt`。这些权重不在 Git 中。本仓库的评估图和表记录了此前完成的实验，但仅凭结果文件无法恢复模型权重。

## 评估

两个评估脚本默认读取各自的最佳权重及全部 200 个测试样本，并同时计算双线性上采样的低分辨率基线。MSE、PSNR、SSIM 只在原始有效网格点上计算。

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_edm.py --steps 18
.\.venv\Scripts\python.exe .\scripts\evaluate_flow.py --steps 20
```

输出分别位于 `experiments/edm_evaluation/` 和 `experiments/flow_evaluation/`，包含逐样本 `metrics.csv`、汇总 `summary.json`、示例图、温度直方图与空间功率谱。要做短流程检查，可在训练时指定 `--max-train-samples 16 --epochs 1 --output <临时权重路径>`，评估时指定 `--checkpoint <临时权重路径> --max-samples 5 --output-dir <临时目录>`。短流程指标不代表完整实验结果。

## 代码位置与来源

数据下载和预处理位于 [`scripts/download_era5.py`](scripts/download_era5.py) 与 [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py)；模型、训练和评估位于同一 `scripts/` 目录。所需的 NVIDIA EDM 代码和许可证见 [`vendor/edm/`](vendor/edm/)。这套 flow matching 代码处理的是 ERA5-Land **气温降尺度**；SatRain 的 VLM 引导降水反演尚未实现。
