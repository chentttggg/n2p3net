# B03 P0 inner producer persistence/replay 规格

日期：2026-10-11。范围是单 participant、至少三个完整且物理独立 C 组的
strict nested calibration-development 软件切片。它不构成公共效能、产品默认、
跨 host 数值包或成人确认。

## 理念与估计量

外层 held 组记为 `v`，候选 selection 层 held 组记为 `h`，grouped-temperature
OOF 的 held 组记为 `t`。一个 inner producer 只能消费

`C \ {v,h,t}` 的训练输入与授权标签；它在 `t` 上的 score 只作为温度 OOF
观测。候选在 `h` 上的证据由 `C \ {v,h}` 训练并通过该组的实际 producer
打分，随后 outer refit 在 `C \ {v}` 上拟合，最终全 C selection/refit 与 outer
metrics 分离。最终选择不能读取任何 `v` score 或 outer metric。

持久化对象必须是实际执行的数值 producer，而不是由 receipt 再拟合得到的
替代品：每个 `FittedEnsemble` 保存其 expert `.pt/.npz`、`ensemble.json`、
完整 training receipts、temperature 数值和 grouped OOF；失败 producer 保存
typed failure receipt。其 fit identity 仍为

`H = SHA256(program, authorized_calibration_receipts, source_checkpoint_digest, checkpoint_role)`。

另外保存独立的 sample bytes digest 和 labels digest，以防 raw receipt 自身未更新
时掩盖 sample mutation。source checkpoint 角色只允许
`pretrained_weights` 或 matched-scratch `architecture_only`，并且保留其实际
SHA-256。

## 信息边界与反例

Recorder 在原 nested run 的 `_SubsetFits.fit`、`_grouped_temperature` OOF fit/score
入口捕获对象和调用上下文。每个 use 保留 `v/h/t` identity 与实际 training IDs；
load/replay 检查这些 forbidden IDs 不在训练 receipt，也不在 OOF held/training
集合中。outer metric 和 final package 没有作为 replay fit 输入。

定向反例包括：

* 改变 h 的 labels/target candidate，独立 labels digest 与 selection scope 校验失败；
* 改变 samples 但故意保留旧 raw_input_digest，独立 sample digest 校验失败；
* 改变 raw half-open scope，scope/input identity 校验失败；
* 修改 manifest fit key 或 ensemble bytes，manifest/package digest 和实际 model
  identity 校验失败；
* 3 组时 grouped temperature 不可用，所有 requested candidate/失败状态仍保留；
  4 组时实际 grouped OOF producers 与每行 expert evidence 均被保存和重放。

默认 replay 使用 exact producer/model identity。`atol`/`rtol` 仅作用于重放 score
记录的浮点字段，程序、scope、fit/source digest 等 identity 仍需精确相等。

## 调用链和接口

1. 调用 `fit_nested_calibration_selection(...,
   inner_replay_recorder=InnerReplayRecorder())`，不改变无 recorder 的现有路径。
2. 原 fit 结束后调用 `persist_inner_replay(selection, calibration, candidates,
   directory, recorder=recorder)`；没有原 run recorder 时明确拒绝持久化。
3. `replay_inner_replay(directory, calibration=..., candidates=...)` 只 load 保存的
   数值包、审计边界并在 held view 上重放 score，不重新 fit。省略 calibration 时
   仍执行包字节、producer identity、denominator 和边界完整性审计。

manifest 具有 schema `n2p3_inner_model_replay/1`，包含完整 candidate roster、
requested denominator、selection decision、outer rows/metrics（仅作为 receipt，
不作为重拟合输入）、每个 producer 的 package、scope、raw/label/sample identities、
source role/digest、uses、selection held score rows 和 OOF rows；`inner_models_persisted=true` 只在上述
同一-run 捕获并成功写出完整包时出现。

## 接受范围与未完成

当前接受：单 participant、至少三组 C、fixed/grouped-temperature candidate roster、
失败/不可用/tie 的 requested denominator、真实 inner 数值包及 OOF score replay、
source role/digest 与 input mutation fail-closed。

仍未完成：B12 `CalibrationPrefix` caller 接入、公共 runner/CLI/BrainSync consumer、
多 participant 层级 nested selection、gating/learned weights、跨 host 同版本整包
复核、公共效能和成人确认。该 slice 也不改变 B12 prefix/K0、spatial/dictionary 或
continuous owner。
