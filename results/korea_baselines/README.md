# Korea baseline comparison

SatRain GMI 韩国测试集，600 个场景。共同有效像素 17,392,665；合格雷达像素 17,393,136；共同覆盖率 99.997292%。评价使用相同场景、雷达参考、网格、质量掩码和降水率阈值。目标配置：`surface_precip`，`min_rqi=0.5`，`min_valid_fraction=0.5`。

`metrics.csv` 为共同像素的误差汇总；`rate_detection.csv` 为按降水率阈值 0.1、10 mm/h 计算的检测结果。CNN 原始输出可能为负；`CNN_clipped` 将负值截为零。GPROF-NN V8 使用外部预训练权重、`ancillary NONE`、float32。CNN 权重来自 Colab 的 batch size 2、FP32 实验（best epoch 18），并非后来在官方 notebook 中以 batch size 32 运行的另一轮训练。

这份目录只保留小型汇总，不含模型权重、原始卫星数据或逐场景预测。原始完整输出在本机的 `experiments/cnn_gprof_korea_comparison/2ad0b10efaaab3ac/`；该目录被 Git 忽略。训练设置和输入差异见根目录 README。
