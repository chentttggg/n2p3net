# 证据深度参数化（D-11 落地）

日期：2026-09-04 · 作者：傻福 · 状态：**已实现，全量测试绿，待云端重评**

## 1. 背景与裁决

`MIN_PROMOTION_HIT_AT_R = 8` 原为硬编码常量，promotion matrix 的 plan 加载层
fail-closed 拒绝一切 `test_reps < 8` 的计划。T 的裁决：

> @8 并不是规范，只是我之前随口说的指标。

→ **证据深度从常量降级为声明式计划参数**。dev 轮可自由选择 R，promotion 深度
另议。

配套的第二条裁决（同日，见 `doc/hit_at_r_valid_denominator_20260904.zh.md`）：

> 逐 R 记录，从 R=2 到 R=8 都记下来，关键是 hit 的计算方法，分母应该只有
> 参与分子判定的那些 valid 被试。

## 2. 设计原则：声明式下放，不放松默认值

| 原则 | 体现 |
|---|---|
| **不声明 = promotion 深度** | 省略 `evidence_floor` 的 plan 行为完全不变（默认 8） |
| **显式才能降低** | 想用浅 R 必须写进 plan / CLI，不存在"默认变松"的路径 |
| **数学硬地板不可破** | `MIN_EVIDENCE_LEVEL = 1`；R < 1 无法支撑任何决策 |
| **自证** | floor 写入 result 的 `target_protocol.minimum_evidence_level`，manifest 层可溯源 |
| **自洽约束** | `1 <= floor <= test_reps`，且 `floor <= 统计设计主 level <= test_reps` |

## 3. 改动清单

| 文件 | 改动 |
|---|---|
| `src/research/contracts.py` | 新增 `MIN_EVIDENCE_LEVEL = 1`；新增 `resolve_evidence_floor()`；`assert_promotion_evidence_gate()` 增加 `minimum_evidence_level` 可选参数（缺省 = `MIN_PROMOTION_HIT_AT_R`）；错误消息改为带实际值 |
| `src/research/__init__.py` | 导出 `MIN_EVIDENCE_LEVEL`、`resolve_evidence_floor` |
| `experiments/run_candidate_promotion_matrix.py` | plan 新增可选字段 `evidence_floor`（进入字段白名单）；`PromotionPlan` 增加该字段；`test_reps` 下界、统计设计主 level 下界、gate 全部改用 floor；子进程 argv 增加 `--min-evidence-level` |
| `experiments/run_candidate_cross_decision.py` | 新增 CLI `--min-evidence-level`（默认 8）；gate 传入 floor；**split 传入 `selection_repetition_floor`**；floor 写入 `target_protocol.minimum_evidence_level` |
| `experiments/analyze_candidate_cross_decision.py` | 从 protocol 读 floor（缺省 8）并传给 gate |
| `experiments/build_candidate_cross_decision_manifest.py` | 从 protocol 读 floor（缺省 8）并传给 gate |
| `src/transfer/candidate_within_subject.py` | **split 门槛与曲线深度解耦**（见 §4）；`_complete_repetition_block()` 增加 `at_least` 开关 |

### 3.1 为什么 floor 写进 `target_protocol` 而不是 `decision`

`decision` 块在 analyze 层是**严格契约相等**对账（`dict(evaluation.decision) != required_decision`
→ 拒绝），它描述"判定机制"，不应随参数漂移。floor 属于"研究协议声明"，
与 `test_repetitions`、`calibration_selections` 同类，放 `target_protocol`。
附带好处：老 result 无此字段 → 缺省 8，**向后兼容**。

## 4. 关键发现：只改下界不够

第一版只做了 plan/CLI 下界参数化，验证时发现**达不到 T 的诉求**。
`candidate_calibration_decision_split` 里两道闸都绑死 `test_repetitions`：

```python
selected = rows[metadata.repetition_indices[rows] < test_repetitions]   # 曲线深度
if not _complete_repetition_block(selected, ..., count=test_repetitions):  # 筛选门槛
```

即：**selection 必须自带 8 个完整 block 才能进账本**。于是

- 设 `test_reps = 8` → 只有 5.2% 的 selection 进得来，但曲线有 R=1..8
- 设 `test_reps = 2` → 68.1% 进得来，但**曲线被砍到 R=1,2**

两条路都拿不到"R=2 到 R=8 逐条记录 + 宽分母"。故必须解耦：

| 概念 | 参数 | 语义 |
|---|---|---|
| **筛选门槛** | `selection_repetition_floor`（= plan 的 `evidence_floor`） | selection 至少携带几个完整 block 才进账本 |
| **曲线深度** | `test_repetitions` | 评估到 R=几，其余记 `incomplete` |

`floor` 缺省等于 `test_repetitions` ⇒ **历史行为逐位不变**。

### 4.1 `_complete_repetition_block` 的隐藏严格相等

解耦时踩到第二个坑：该函数用 `len(rows) != 12 * count` 判定，语义是
"**恰好** count 个 block"。于是 rep=3 的 selection 在 `count=2` 下被拒（36 ≠ 24）。

该函数共 3 处调用，其中两处（校准校验 L176、时序 fold 的 pre/post L290/304）
传入的是**已按 rep 区间精确切片**的 rows，严格相等天然成立。故增加
`at_least: bool = False` 开关，**默认保持严格相等**，仅 split 门槛处传 `True`。

## 5. 测试覆盖（新增 15 项）

契约层（`tests/test_research_contracts.py`）：
- 缺省 = promotion 深度；`< 1` 与非整数被拒
- gate 接受声明的浅 horizon；`test_reps < floor` 被拒
- 主 level 必须落在 `[floor, test_reps]`；`evidence_levels` 必须含 floor
- **未声明时仍要求含 hit@8**（回归保护）

计划层（`tests/test_candidate_promotion_matrix.py`）：
- 省略 `evidence_floor` → `plan.evidence_floor == 8`
- 声明 floor=2 + test_reps=2 + level=2 → 通过
- floor ∈ {0, -3} / {True, 2.5, "2"} 被拒
- `test_reps < floor`、`level > test_reps` 被拒

Split 层（`tests/test_candidate_transfer.py`）：
- **解耦核心用例**：8 个 selection 携带 1/2/3/4/8 个 rep，校准 3 个
  - 历史行为（floor=8）：仅 rep=8 的 selection 进账本，其余 4 个记
    `insufficient_complete_test_repetitions`
  - floor=2：rep >= 2 的 4 个全部进入，**且最深 selection 的 8×12 行全部保留**
    （证明曲线深度未受损）
- floor 缺省 == 显式传 `test_repetitions`
- floor ∈ {0,-1} / {True, 2.0, "2"} / `> test_reps` 被拒

## 6. 向后兼容性

| 对象 | 影响 |
|---|---|
| 已冻结的 plan（无 `evidence_floor`） | 行为逐位不变 |
| 已产出的 result（protocol 无 `minimum_evidence_level`） | analyze / manifest 缺省 8，行为不变 |
| 云端 r1/r2/r3 产物 | 全部可用（跑在冻结树 9f0b421e） |
| 已生成的 archive / manifest 摘要 | 不受影响（新字段只出现在新产物） |

## 7. 下一步：浅 R 重评（待 r3 结束后执行）

r3（E=16）跑完即获得最深 checkpoint。用新参数**只重评 result 层**（不重训）：

| 参数 | 值 | 理由 |
|---|---|---|
| `evidence_floor` | 2 | rep>=2 占 68.1%（vs rep>=8 的 5.2%），selection 通过率约 ×5.7 |
| `test_reps` | 8 | 保留 R=1..8 完整曲线 |
| 统计设计主 level | 待定（建议 8） | 与 r1/r2 曲线可比 |
| epochs | 沿用 r3（16） | 单变量纪律：本轮只改证据深度 |

预期收益（按 rep 分布外推）：

| | 现状（floor=8） | floor=2 |
|---|---|---|
| test selection 通过率 | 5.2% | ≈68% |
| 有效被试 | 31/64 | ≈64/64 |
| 判定总数 | 73 | ≈420 |
| 被试级 SE | 0.085 | ≈0.045 |

**注意**：此表为按 rep 分布的外推，非实测。实测值以重评结果为准。
