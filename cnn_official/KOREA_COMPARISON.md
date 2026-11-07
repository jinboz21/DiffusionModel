# CNN / GPROF-NN 韩国共同像素对照

在项目根目录 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -m cnn_official.compare_korea
```

该命令只进行 CNN 推理和指标计算，不训练、不下载。默认使用已找回的
`colab_gmi_20261105_221329/checkpoints/best-epoch=18.ckpt`，以及
`experiments/gprof_korea/full/evaluation/gridded` 中已保存的 GPROF-NN 结果。
这份 CNN 权重来自 batch size 2、FP32 的 Colab 训练；不是后来 batch size 32 的另一轮训练权重。

之前 notebook 完整评估没有保存逐场景预测，所以需要重新进行一次 CNN 推理。
脚本使用 CPU、FP32、batch size 2、tile size 256、overlap 64，逐场景缓存预测。
中断后重新运行相同命令，会复用已完成的缓存；不必重新训练。

结果写入 `experiments/cnn_gprof_korea_comparison/<配置指纹>/`：

- `metrics.csv`：共同像素上的 MAE、RMSE、绝对 Bias（mm/h）和相对 Bias（%）。
- `rate_detection.csv`：统一按降水率 >= 0.1 和 >= 10 mm/h 计算 CSI、POD、FAR。
- `coverage.csv`：每个场景的雷达有效像素、两模型有效像素和交集数量。
- `provenance.json`：权重哈希、代码哈希、输入配置及依赖版本。
- `completion.json`：全部 600 场景完成后才生成。
- `cnn_predictions/`：逐场景原始 CNN 降水率，供续跑及后续分析使用。

共同掩码使用 GPROF 评估保存的雷达质量条件、卫星覆盖范围，并取两个模型有限预测的交集。
每个模型的指标都在完全相同的像素上累积，不对场景均值简单取平均。
脚本核对时间戳、网格、参考降水及 GPROF 掩码来源，发现不匹配会停止。

`CNN_raw` 保留原始负预测；`CNN_clipped` 将负值截为零，并单独标识这种后处理。
GPROF 和全零参考也在同一交集上计分。降水检测表使用降水率阈值，
不能直接与之前 CNN 概率头按 0.5 阈值计算的检测表混用。

这是同一测试集上的方法对照。CNN 在 SatRain xs 上训练；GPROF-NN 使用外部预训练权重，
因此不能称为“两个模型在相同训练数据和训练预算下的对照”。

这套对照已由用户运行完 600 个场景；小型汇总结果见根目录 `results/korea_baselines/`。
