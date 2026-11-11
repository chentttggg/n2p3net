# 早期小运行文字记录汇编

本页提取原运行记录中的任务、数量、维度、汇总指标和性能读数。原始 EEG、参与者逐行记录、逐试次预测及训练数组未收录。原文件 SHA-256 用于绑定摘录来源，运行旧字段保留原口径。

[返回参考资料索引](README.zh.md)

<a id="h06"></a>

## H06：v8_audit_loso_smoke

```json
{
  "source_record_sha256": "ae6f95510fc18208bd4ced1b63a2c2934386ce544e080f52c43156805f1d98e6",
  "mode": "evaluate",
  "data_summary": {
    "n_subjects": 3,
    "n_trials": 463,
    "target_rate": 0.09719222462203024,
    "X3_shape": [
      463,
      3,
      358
    ],
    "X8_shape": [
      463,
      3,
      358
    ]
  },
  "results": {
    "hit_rate_mean": 1.0,
    "hit_rate_std": 0.0,
    "balanced_acc_mean": 0.6379144305605617,
    "auc_mean": 0.7230358422221332,
    "transductive_balanced_acc_mean": 0.6343252577640719,
    "primary_decision_metric": "llr@15"
  },
  "decision_metrics": {
    "sum@1": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 1,
      "aggregation": "sum",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "mean@1": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 1,
      "aggregation": "mean",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "llr@1": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 1,
      "aggregation": "llr",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "sum@3": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 3,
      "aggregation": "sum",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "mean@3": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 3,
      "aggregation": "mean",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "llr@3": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 3,
      "aggregation": "llr",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "sum@5": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 5,
      "aggregation": "sum",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "mean@5": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 5,
      "aggregation": "mean",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "llr@5": {
      "hit_rate": 0.3333333333333333,
      "evidence_budget": 5,
      "aggregation": "llr",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "sum@10": {
      "hit_rate": 0.5,
      "evidence_budget": 10,
      "aggregation": "sum",
      "n_covered": 2,
      "n_total": 3,
      "coverage": 0.6666666666666666
    },
    "mean@10": {
      "hit_rate": 0.5,
      "evidence_budget": 10,
      "aggregation": "mean",
      "n_covered": 2,
      "n_total": 3,
      "coverage": 0.6666666666666666
    },
    "llr@10": {
      "hit_rate": 0.5,
      "evidence_budget": 10,
      "aggregation": "llr",
      "n_covered": 2,
      "n_total": 3,
      "coverage": 0.6666666666666666
    },
    "sum@15": {
      "hit_rate": 1.0,
      "evidence_budget": 15,
      "aggregation": "sum",
      "n_covered": 2,
      "n_total": 3,
      "coverage": 0.6666666666666666
    },
    "mean@15": {
      "hit_rate": 1.0,
      "evidence_budget": 15,
      "aggregation": "mean",
      "n_covered": 2,
      "n_total": 3,
      "coverage": 0.6666666666666666
    },
    "llr@15": {
      "hit_rate": 1.0,
      "evidence_budget": 15,
      "aggregation": "llr",
      "n_covered": 2,
      "n_total": 3,
      "coverage": 0.6666666666666666
    },
    "sum@all": {
      "hit_rate": 1.0,
      "evidence_budget": "all",
      "aggregation": "sum",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "mean@all": {
      "hit_rate": 1.0,
      "evidence_budget": "all",
      "aggregation": "mean",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    },
    "llr@all": {
      "hit_rate": 1.0,
      "evidence_budget": "all",
      "aggregation": "llr",
      "n_covered": 3,
      "n_total": 3,
      "coverage": 1.0
    }
  },
  "protocol_summary": {
    "val_subject_frac": null,
    "early_stop_patience": 6,
    "last_fold_val_subjects": 0,
    "last_fold_n_val_epochs": 0,
    "erp_calibration": "fixed",
    "last_fold_erp_calibration": null
  }
}
```

<a id="h07"></a>

## H07：v8_timing_benchmark_bs512_20260824

```json
{
  "source_record_sha256": "593b73d666fb2f4abca26650cf14c6b0978d6ece1aa32a2ebc70547a565c5727",
  "mode": "benchmark",
  "data_summary": {
    "n_subjects": 242,
    "n_trials": 36625,
    "target_rate": 0.1138839590443686,
    "X3_shape": [
      36625,
      3,
      358
    ],
    "X8_shape": [
      36625,
      3,
      358
    ]
  },
  "benchmark_summary": {
    "n_train_trials": 36431,
    "epochs": 3,
    "wall_seconds": 31.465373200000613,
    "seconds_per_epoch": 10.488457733333538,
    "gpu_peak_memory_mb": 1629.638144
  },
  "protocol_summary": {}
}
```

<a id="h08"></a>

## H08：v8_joint_smoke

```json
{
  "source_record_sha256": "e5abec796e3e7786837624ff7c5eaceff64edfa7beddcdec4c42155b0dd67874",
  "mode": "benchmark",
  "data_summary": {
    "n_subjects": 3,
    "n_trials": 463,
    "target_rate": 0.09719222462203024,
    "X3_shape": [
      463,
      3,
      358
    ],
    "X8_shape": [
      463,
      3,
      358
    ]
  },
  "benchmark_summary": {
    "n_train_trials": 269,
    "epochs": 3,
    "wall_seconds": 4.5901319999975385,
    "seconds_per_epoch": 1.5300439999991795,
    "gpu_peak_memory_mb": 518.799872
  },
  "protocol_summary": {}
}
```

<a id="h09"></a>

## H09：v8_long_smoke

```json
{
  "source_record_sha256": "3debec3432969c4659298f7c88170b98d3ba2f8c6e415f32fe44f1d154d3d638",
  "mode": "benchmark",
  "data_summary": {
    "n_subjects": 3,
    "n_trials": 463,
    "target_rate": 0.09719222462203024,
    "X3_shape": [
      463,
      3,
      358
    ],
    "X8_shape": [
      463,
      3,
      358
    ]
  },
  "benchmark_summary": {
    "n_train_trials": 269,
    "epochs": 1,
    "wall_seconds": 5.125227900000027,
    "seconds_per_epoch": 5.125227900000027,
    "gpu_peak_memory_mb": 519.973376
  },
  "protocol_summary": {}
}
```

<a id="h10"></a>

## H10：v8_timing_benchmark_20260824

```json
{
  "source_record_sha256": "7e46d96395479d1cb90c04625efd3b848ab7796cfa5f4799afa10129e8f65370",
  "mode": "benchmark",
  "data_summary": {
    "n_subjects": 242,
    "n_trials": 36625,
    "target_rate": 0.1138839590443686,
    "X3_shape": [
      36625,
      3,
      358
    ],
    "X8_shape": [
      36625,
      3,
      358
    ]
  },
  "benchmark_summary": {
    "n_train_trials": 36431,
    "epochs": 3,
    "wall_seconds": 37.28157780000038,
    "seconds_per_epoch": 12.427192600000126,
    "gpu_peak_memory_mb": 1629.638144
  },
  "protocol_summary": {}
}
```

<a id="h11"></a>

## H11：neural_ride_v7_smoke

```json
{
  "source_record_sha256": "bfede5f3f4505eb6caa5208a6b62f86a8643e833a751ca99a80dc063488e8562",
  "mode": "benchmark",
  "data_summary": {
    "n_subjects": 3,
    "n_trials": 466,
    "target_rate": 0.09656652360515021,
    "X3_shape": [
      466,
      3,
      256
    ],
    "X8_shape": [
      466,
      3,
      256
    ]
  },
  "benchmark_summary": {
    "n_train_trials": 271,
    "epochs": 1,
    "wall_seconds": 4.544264000000112,
    "seconds_per_epoch": 4.544264000000112,
    "gpu_peak_memory_mb": 201.772032
  },
  "protocol_summary": {}
}
```

<a id="h12"></a>

## H12：neural_ride_v7_smoke_final

```json
{
  "source_record_sha256": "e403dff3353ec96abb97898ec66688960f5335d0ce14d7dde4129b614f71b515",
  "mode": "benchmark",
  "data_summary": {
    "n_subjects": 3,
    "n_trials": 466,
    "target_rate": 0.09656652360515021,
    "X3_shape": [
      466,
      3,
      256
    ],
    "X8_shape": [
      466,
      3,
      256
    ]
  },
  "benchmark_summary": {
    "n_train_trials": 271,
    "epochs": 1,
    "wall_seconds": 4.220542299999579,
    "seconds_per_epoch": 4.220542299999579,
    "gpu_peak_memory_mb": 264.729088
  },
  "protocol_summary": {}
}
```

<a id="h13"></a>

## H13：neural_ride_v7_smoke_multiset

```json
{
  "source_record_sha256": "004308df2347345fe7460031bac20c72d0a4671bfbd110101d75ddb2bb2606b3",
  "mode": "benchmark",
  "data_summary": {
    "n_subjects": 3,
    "n_trials": 466,
    "target_rate": 0.09656652360515021,
    "X3_shape": [
      466,
      3,
      256
    ],
    "X8_shape": [
      466,
      3,
      256
    ]
  },
  "benchmark_summary": {
    "n_train_trials": 271,
    "epochs": 1,
    "wall_seconds": 3.738600699998642,
    "seconds_per_epoch": 3.738600699998642,
    "gpu_peak_memory_mb": 264.728064
  },
  "protocol_summary": {}
}
```
