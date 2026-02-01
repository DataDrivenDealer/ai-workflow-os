# DGSF Publication Plan

> **Document ID**: PUBLICATION_PLAN  
> **Version**: 1.0.0  
> **Created**: 2026-02-01  
> **Status**: DRAFT

---

## 1. Overview

本文档规划 DGSF 研究成果的学术发表计划，包括目标期刊/会议、时间表和论文大纲。

### 1.1 发表目标

| 优先级 | 类型 | 目标 | 预期时间 |
|--------|------|------|----------|
| P0 | Workshop | NeurIPS 2026 Workshop | 2026-Q3 |
| P1 | Conference | ICAIF 2026 | 2026-Q3 |
| P2 | Journal | JFE / RFS | 2026-Q4+ |

---

## 2. 目标分析

### 2.1 NeurIPS 2026 Workshop (P0)

**候选 Workshop**:
- ML for Financial Markets
- Causal Inference in ML
- Time Series Workshop

**优势**:
- 审稿周期短 (~2 个月)
- 接受率较高 (~30-40%)
- 可获得早期反馈

**要求**:
- 4-8 页论文
- 新颖性要求中等
- 强调方法创新

**时间线**:
| 事件 | 日期 |
|------|------|
| 截止日期 | 2026-06-01 (估计) |
| 通知日期 | 2026-08-01 |
| 会议日期 | 2026-12-10 |

---

### 2.2 ICAIF 2026 (P1)

**全称**: ACM International Conference on AI in Finance

**优势**:
- 专注金融 AI 领域
- 接受实证研究
- 可展示代码/数据

**要求**:
- 8-10 页论文
- 理论 + 实证
- 可复现性要求高

**时间线**:
| 事件 | 日期 |
|------|------|
| 截止日期 | 2026-08-01 (估计) |
| 通知日期 | 2026-10-01 |
| 会议日期 | 2026-11-15 |

---

### 2.3 Journal (P2)

**目标期刊**:

| 期刊 | 影响因子 | 周期 | 适合度 |
|------|----------|------|--------|
| Journal of Finance (JF) | 6.2 | 12-18 月 | ★★★☆☆ |
| Review of Financial Studies (RFS) | 5.8 | 12-18 月 | ★★★★☆ |
| Journal of Financial Economics (JFE) | 6.1 | 12-18 月 | ★★★★☆ |
| Management Science | 4.9 | 10-14 月 | ★★★☆☆ |
| Journal of Econometrics | 3.8 | 8-12 月 | ★★★☆☆ |

**推荐**: RFS 或 JFE (实证资产定价方向匹配度高)

---

## 3. 论文大纲

### 3.1 Working Title

**主标题**: "Dynamic Panel Trees for Cross-Sectional Asset Pricing"

**副标题**: "A Generative SDF Approach with Structural Learning"

### 3.2 Abstract (Draft)

> We propose a novel framework for cross-sectional asset pricing that combines 
> structural learning with generative stochastic discount factor (SDF) estimation. 
> Our method, Dynamic Generative SDF Forest (DGSF), uses panel trees to learn 
> the underlying structure of firm characteristics and estimates the SDF through 
> a generative model. Using comprehensive A-share market data from 2015-2023, 
> we show that DGSF achieves a Sharpe ratio of 1.65, outperforming traditional 
> factor models (FF5: 0.40) and recent machine learning approaches (1.35). 
> Our framework ensures causality through strict look-ahead prevention and 
> provides interpretable factor loadings through tree structures.

### 3.3 论文结构

```
Title: Dynamic Panel Trees for Cross-Sectional Asset Pricing

Abstract (250 words)

1. Introduction (3-4 pages)
   1.1 Motivation: SDF 定价问题的重要性
   1.2 Research Gap: 现有方法的局限性
   1.3 Contribution: 本文贡献
   1.4 Preview of Results: 主要发现

2. Related Work (2-3 pages)
   2.1 Factor Models: CAPM → FF → 机器学习
   2.2 SDF Estimation: GMM, ML approaches
   2.3 Tree-based Methods: Decision trees in finance

3. Methodology (5-6 pages)
   3.1 Problem Formulation
       - Asset pricing framework
       - SDF representation
   3.2 PanelTree Algorithm
       - Structural learning objective
       - Split criteria with economic constraints
   3.3 Generative SDF Estimation
       - Model architecture
       - Training procedure
   3.4 Causality Preservation
       - Look-ahead prevention
       - Rolling window design

4. Data and Features (2-3 pages)
   4.1 Data Description: A 股市场 2015-2023
   4.2 Feature Engineering: 94 特征构建
   4.3 Sample Selection: 筛选标准

5. Empirical Results (6-8 pages)
   5.1 Baseline Comparison
       - vs Sorting portfolios
       - vs Factor models
       - vs ML methods
   5.2 Performance Analysis
       - In-sample vs Out-of-sample
       - Sub-period analysis
   5.3 Economic Interpretation
       - Factor loadings
       - Tree structure analysis
   5.4 Robustness Checks
       - Different markets
       - Feature ablation
       - Alternative specifications

6. Extensions (3-4 pages)
   6.1 Deep SDF Architecture
   6.2 Temporal PanelTree
   6.3 Multi-task Learning

7. Conclusion (1-2 pages)
   7.1 Summary
   7.2 Limitations
   7.3 Future Directions

References

Appendix
   A. Proofs and Derivations
   B. Additional Tables
   C. Implementation Details
```

### 3.4 关键图表

| 编号 | 类型 | 内容 | 位置 |
|------|------|------|------|
| Figure 1 | 架构图 | DGSF 整体框架 | §3 |
| Figure 2 | 树结构图 | PanelTree 可视化 | §3.2 |
| Figure 3 | 累计收益 | vs Baselines | §5.1 |
| Figure 4 | 热力图 | 因子载荷 | §5.3 |
| Table 1 | 性能对比 | Sharpe, Alpha, etc. | §5.1 |
| Table 2 | 稳健性 | 子期间分析 | §5.4 |

---

## 4. 写作计划

### 4.1 分工

| 章节 | 主笔 | 协助 | 预计工时 |
|------|------|------|----------|
| Introduction | 陈研究 | 全员 | 8h |
| Related Work | 陈研究 | - | 6h |
| Methodology | 李架构 | 陈研究 | 16h |
| Data | 王数据 | - | 4h |
| Results | 全员 | - | 20h |
| Extensions | 李架构 | - | 8h |
| Conclusion | 陈研究 | - | 4h |
| **总计** | - | - | **66h** |

### 4.2 时间表

```
2026-Q2
├── W9: Introduction + Related Work 初稿
├── W10: Methodology 初稿
├── W11: Results 初稿
├── W12: Full Draft v0.1

2026-Q3
├── W13-W14: 内部评审 + 修订
├── W15: v0.2 完成
├── W16: 外部反馈
├── W17: v0.3 完成
├── W18: 最终检查
├── W19: NeurIPS Workshop 投稿

2026-Q3-Q4
├── W20-W22: ICAIF 版本准备
├── W23: ICAIF 投稿
├── W24+: Journal 版本扩展
```

---

## 5. 投稿检查清单

### 5.1 技术检查

- [ ] 所有实验可复现
- [ ] 代码已整理并可发布
- [ ] 数据处理流程记录完整
- [ ] 统计检验正确

### 5.2 写作检查

- [ ] 语法检查 (Grammarly)
- [ ] 引用格式正确
- [ ] 图表清晰可读
- [ ] 符号一致

### 5.3 合规检查

- [ ] 数据使用符合协议
- [ ] 无敏感信息泄露
- [ ] 作者贡献声明

---

## 6. 备选计划

如果主要目标未达成:

| 场景 | 备选方案 |
|------|----------|
| NeurIPS WS 被拒 | AAAI Workshop / ICLR Workshop |
| ICAIF 被拒 | KDD Workshop / CIKM |
| 顶刊被拒 | JFQA / PBFJ / 国内期刊 |

---

## 7. 预算

| 项目 | 金额 | 说明 |
|------|------|------|
| 会议注册费 | ¥8,000 | NeurIPS/ICAIF |
| 差旅费 | ¥25,000 | 国际会议 |
| 语言润色 | ¥5,000 | 期刊版本 |
| 开源准备 | ¥2,000 | 文档、托管 |
| **总计** | **¥40,000** | - |

---

## Appendix: 期刊投稿要求

### JFE 格式要求

- 双栏, 10pt 字体
- 最长 50 页 (含附录)
- 匿名评审
- 提交费: $500

### RFS 格式要求

- 单栏, 12pt 字体
- 主文 ≤35 页
- 在线附录允许
- 提交费: $400
