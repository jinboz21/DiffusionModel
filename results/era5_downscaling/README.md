# ERA5-Land temperature downscaling results

本目录保存一项**独立于 SatRain 降水反演**的 ERA5-Land 2 米气温降尺度实验结果。输入为低分辨率气温场，参考为高分辨率 ERA5-Land 气温场。EDM diffusion 与 conditional flow matching 均在相同的 200 个测试样本（`sample_2200` 至 `sample_2399`）上评估；数据按时间顺序分为 2,000 个训练、200 个验证和 200 个测试样本。

| 方法 | 平均 MSE (K²) ↓ | 平均 PSNR (dB) ↑ | 平均 SSIM ↑ | 采样步数 |
| --- | ---: | ---: | ---: | ---: |
| 上采样低分辨率输入 | 1.7216 | 35.5182 | 0.9009 | — |
| EDM diffusion | 0.5875 | 40.3855 | 0.9540 | 18 |
| Conditional flow matching | 0.5725 | 40.5677 | 0.9569 | 20 |

表中指标是先对每个测试样本计算，再对 200 个样本取平均。PSNR 使用相同的 77.9011 K 数据范围。两个生成模型使用不同的采样步数，表格记录的是本次实验结果，不代表严格匹配计算预算的比较。

- [`edm/`](edm/)：从本机 `experiments/edm_preview/` 复制的 200 样本结果。
- [`flow_matching/`](flow_matching/)：从本机 `experiments/flow_evaluation_full/` 复制的 200 样本结果。

每个目录中的 `metrics.csv` 为逐样本指标，`summary.json` 为平均指标及评估设置；`example.png`、`histogram.png` 和 `power_spectrum.png` 分别展示示例气温场、数值分布和空间功率谱。这里没有模型权重或原始 ERA5-Land 数据。
