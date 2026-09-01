# 合同加固的外部依据

本轮为定向检索，不是系统综述。检索时间为 2026-09-01；Parallel 与 Exa 后端均因
本机无认证不可用，论文由 OpenAlex/Crossref 公共 API 定向获取，规范直接读取官方页。
原始响应保存在忽略目录 `tmp/contract_research_20260901/`。

- BIDS 将 participant、session、task、run 分开建模；本项目进一步明确
  `Session != Decision != Block != StimulusEvent`。v2 BrainSync block 不能推断 decision。
  参考：Gorgolewski et al., 2016, DOI `10.1038/sdata.2016.44`；BIDS Common Principles。
- BIDS Derivatives 与 W3C PROV-DM 要求派生实体保留来源/活动关系。因此 identity table
  只负责参与者同一性，DataLineage 单独记录 parent entity、operation 和参数 digest。
- Signal alignment for cross-datasets in P300 BCIs 支持显式对齐而非静默通道替代，
  DOI `10.1088/1741-2552/ad430d`。这不证明本项目的 common-CAR 会提升性能。
- Model Cards（DOI `10.1145/3287560.3287596`）与 Datasheets for Datasets
  （DOI `10.1145/3458723`）支持把用途、来源、限制和评估范围绑定到机器可读制品。
- Varma & Simon 2006（DOI `10.1186/1471-2105-7-91`）、Varoquaux 2018
  （DOI `10.1016/j.neuroimage.2017.06.061`）和 Bates et al. 2023
  （DOI `10.1080/01621459.2023.2197686`）支持区分模型选择偏差、小样本误差条和
  cross-validation 的条件 estimand。因此固定 checkpoint 的 participant interval 与
  training-procedure interval 分开实现。
- The Identity Trap in EEG Foundation Models（arXiv:2606.06647）是 2026 预印本，
  只用于设计 subject-identity 反例，不作为已确认性能结论。
