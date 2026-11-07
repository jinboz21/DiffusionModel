# GPROF-NN V8 on the official SatRain Korea test scenes

此目录提供适配代码，由用户运行。单场景检查和韩国 600 场景的共同像素对照均已完成；
复现时请先做单场景检查，再运行完整测试。

## 实验定义

- 使用官方 CNN Notebook 最后评价的 `gmi / testing / korea`，不是 82 个 validation 样本。
- 测试场景清单来自已安装 SatRain 的官方 manifest，脚本打印实际数量。
  `xs` 是训练/验证子集设置，不用于韩国测试集的筛选。
- CNN 接受 gridded 输入；这里的预训练 GPROF-NN 接受同一次过境的 on_swath 原始亮温。
  SatRain 官方 Evaluator 使用 target 的 scan_index/pixel_index 将结果映射到共同的 gridded 雷达标签。
  相同评价场景并不意味着两者的输入张量/空间采样完全相同。
- 下载每场景三个文件：on_swath GMI、on_swath target、gridded target。
  target 中的经纬度仅用于定位与官方质量检查；降水标签不送入模型。
- 本配置为 **GPROF-NN V8 pretrained / ancillary NONE / float32**。
  14 个辅助变量通道使用 NaN，与安装版 GPROF-NN 的无预处理器 L1C 路径一致。
  使用传感器名义入射角和官方输出后处理。没有 land_fraction，不能获得完整的陆海像素偏差修正；
  官方按观测日期进行的 boost 修正仍由原代码处理。没有辅助信息时，冻结降水等附加变量不应使用。
- 这是外部预训练参考基线，不能称为 2022 版 GPROF-NN 或完整 CLI/STD 辅助配置的官方复现。
  后续若要完整辅助数据实验，必须另外准备兼容的官方预处理数据并建立独立实验目录。
- 评价输出的降水/强降水识别来自降水率 >=0.1、>=10 mm/h；没有将概率头混入这组指标。
  与 CNN 比较时也应使用其降水率阈值结果，概率头结果另列。

## 先运行一个场景（项目根目录 PowerShell）

此前运行使用独立的 Python 3.12 环境、GPROF-NN `0.8.21.dev2+ge27d590a5`，源码提交为 `e27d590a50bf1b65c53567b872d7b222b161ceb6`（`gprof_v8` 分支）。新环境可先安装该版本并下载预训练 GMI 模型：

```powershell
py -3.12 -m venv .venv_gprof_v8
.\.venv_gprof_v8\Scripts\python.exe -m pip install "numpy>=2.4,<2.5" "git+https://github.com/simonpf/gprof_nn@e27d590a50bf1b65c53567b872d7b222b161ceb6"
.\.venv_gprof_v8\Scripts\gprof_nn.exe download model gmi
```

SatRain 数据准备与评价使用根目录 README 中的 `.venv`，GPROF-NN 推理使用 `.venv_gprof_v8`。下列命令中模型路径是 Windows 的默认缓存位置；其他系统请传入实际模型路径。

```powershell
.\.venv\Scripts\python.exe -m gprof_korea.prepare --limit 1

.\.venv_gprof_v8\Scripts\python.exe -m gprof_korea.infer --model "$env:LOCALAPPDATA\gprof_nn\gprof_nn\models\gprof_nn_3d_gmi.pt" --device cpu

.\.venv\Scripts\python.exe -m gprof_korea.evaluate
```

按顺序执行，前一步报错就停止并保留完整错误输出。已有环境和模型时无需重复安装或下载。
prepare 默认只下载第一个场景，并打印文件维度与实际磁盘大小。
infer 默认只推理 manifest 的第一个场景，保留官方 checkpoint 推理配置与归一化。
输入若不是 `(13, >=100 scans, 221 pixels)`，或时间/坐标不匹配，会明确报错，避免猜测格式。
不要将之前已归一化的 CNN 张量代替原始亮温。

## 单场景检查通过后，完整运行

```powershell
.\.venv\Scripts\python.exe -m gprof_korea.prepare --all

.\.venv_gprof_v8\Scripts\python.exe -m gprof_korea.infer --model "$env:LOCALAPPDATA\gprof_nn\gprof_nn\models\gprof_nn_3d_gmi.pt" --device cpu --all --output experiments/gprof_korea/full/predictions

.\.venv\Scripts\python.exe -m gprof_korea.evaluate --all --predictions experiments/gprof_korea/full/predictions --output experiments/gprof_korea/full/evaluation
```

必须等 prepare 打印 `PREPARATION COMPLETE: 600/600 official scenes indexed.`（数量以官方清单为准）
并正常退出，再启动推理。下载中断时重跑 prepare 会补齐缺失文件；旧的单场景清单在准备成功前保持不变。
全量运行使用 `full/` 子目录，保留之前的单场景结果，也避免脚本版本更新后的签名冲突。
若该环境的 PyTorch 支持 CUDA，可将 `--device cpu` 改为 `--device cuda`。
prepare 复用已下载文件；infer 仅在模型和适配代码签名一致时复用已有输出。
完整下载的 GB 数没有预先测定，脚本完成后报告实际值，不用训练 xs 数据量估算。

## 输出

- `experiments/gprof_korea/manifest.json`：选定的官方场景、文件路径和完整场景数。
- `predictions/`：逐场景原始沿轨输出及版本、模型 SHA256 配置。
- `evaluation/metrics.csv`：MAE、RMSE、Bias；零降水对照使用完全相同的像素。
- `evaluation/rate_detection.csv`：CSI、POD、FAR 与 TP/FP/FN。
- `evaluation/official_*.csv`：SatRain 官方评价器统计。
- `evaluation/coverage.csv` 和 `summary.json`：参与评价的像素数、缺失预测数、是否完整测试。
- `evaluation/gridded/paired_*.nc`：对齐后的预测、参考数据、common_valid_mask。

完整比较需要重新对 CNN 使用相同场景和 common_valid_mask 计算指标。
若 CNN 也有非有限输出，还要进一步取两个模型有效像素的交集。
不得直接把之前的 validation 成绩或不同掩码的韩国成绩并列表述为公平比较。

参考实现来自本机安装的 `gprof_nn.retrieval.load_input_data_l1c`、
`GPROFNNInputLoader.finalize_results`、`pytorch_retrieve.inference.SequentialInferenceRunner`
和 `satrain.evaluation.Evaluator`。
