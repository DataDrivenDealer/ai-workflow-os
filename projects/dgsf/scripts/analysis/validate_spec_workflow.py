"""
VS Code + Copilot 交互式验证场景
=================================

本脚本提供了一系列可在 VS Code 中与 Copilot 交互验证的场景。
每个场景模拟真实的量化开发问题，触发 Skills + MCP + Hooks 工作流。

使用方法:
---------
1. 在 VS Code 中打开此文件
2. 使用 Copilot Chat，输入对应的触发命令
3. 观察 Copilot 如何调用 Skills 和 MCP Tools

验证检查清单:
-------------
□ Skills 被正确调用
□ MCP Tools 返回预期结果
□ Hooks 在适当时机触发
□ 工作流完整闭环
"""

import sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# 场景 1: OOS Sharpe 低于阈值 → Spec 演进
# ============================================================================

SCENARIO_1_PROBLEM = """
实验 t05_momentum 的结果显示：
- OOS Sharpe = 0.8 (低于阈值 1.5)
- IS Sharpe = 1.6
- OOS/IS Ratio = 0.5 (严重过拟合)

当前 SDF_INTERFACE_CONTRACT.yaml 中的 min_sharpe_threshold 设置为 1.0，
这允许了性能不佳的模型通过验证。

问题：阈值是否应该调整？如果是，应该改为多少？
"""

# 触发命令（在 Copilot Chat 中输入）：
# /dgsf_spec_triage 实验 t05_momentum OOS Sharpe = 0.8，低于行业标准 1.5

# 预期工作流：
# 1. spec_triage → 分类为 metric_deviation / spec_issue
# 2. /dgsf_research → 研究最佳阈值
# 3. /dgsf_spec_propose → 生成变更提案
# 4. 人工审批
# 5. /dgsf_spec_commit → 提交变更


# ============================================================================
# 场景 2: 接口不兼容 → Spec 设计问题
# ============================================================================

SCENARIO_2_PROBLEM = """
代码审查发现 PanelTree 模块与 SDF 模块的接口不一致：

PanelTree 输出格式:
    output = {"features": tensor, "mask": tensor}

SDF 期望格式:
    input = {"X": tensor, "valid_mask": tensor}

这导致集成测试失败：
    KeyError: 'X' in sdf/model.py line 42

接口契约 PANELTREE_SDF_INTERFACE 中未明确定义键名规范。
"""

# 触发命令：
# /dgsf_spec_triage 接口不兼容，PanelTree 输出 features 但 SDF 期望 X

# 预期结果：
# - 分类: design_issue / spec_issue
# - 推荐: /dgsf_research → /dgsf_spec_propose


# ============================================================================
# 场景 3: 数据验证规则过严 → 实验无法运行
# ============================================================================

SCENARIO_3_PROBLEM = """
数据验证规则阻止了新数据集的加载：

ValidationError: Date range 2020-01-01 to 2025-12-31 exceeds 
                 max_date_range of 3 years in DATA_QUALITY_STANDARD.yaml

但对于长期因子研究，我们需要至少 5 年的数据才能进行稳健性检验。

当前配置:
    data_governance:
      max_date_range_years: 3  # 过于严格

建议修改为 10 年。
"""

# 触发命令：
# /dgsf_spec_triage 数据验证规则 max_date_range 过严，阻止加载 5 年数据

# 预期结果：
# - 分类: runtime_error / spec_issue
# - 影响: DATA_QUALITY_STANDARD.yaml


# ============================================================================
# 场景 4: 代码 Bug（非 Spec 问题）
# ============================================================================

SCENARIO_4_PROBLEM = """
运行 pytest 时出现错误：

TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'
  File "sdf/loss.py", line 78, in compute_loss
    total = base_loss + regularization
    
这是代码逻辑错误，regularization 在某些情况下为 None。
"""

# 触发命令：
# /dgsf_spec_triage TypeError NoneType + float in sdf/loss.py line 78

# 预期结果：
# - 分类: runtime_error / code_bug
# - 推荐: /dgsf_diagnose（不是 spec 变更）


# ============================================================================
# 验证辅助函数
# ============================================================================

def print_scenario(scenario_num: int, problem: str):
    """打印场景信息"""
    print(f"\n{'='*60}")
    print(f"场景 {scenario_num}")
    print('='*60)
    print(problem)
    print('-'*60)


def simulate_triage_call(problem_description: str, source: str = "experiment"):
    """
    模拟 spec_triage MCP 调用
    
    在实际使用中，这会通过 MCP Server 调用。
    此处仅用于演示预期行为。
    """
    print(f"\n[MCP CALL] spec_triage")
    print(f"  problem: {problem_description[:80]}...")
    print(f"  source: {source}")
    
    # 简单的关键词匹配（实际逻辑在 MCP Server 中）
    keywords = problem_description.lower()
    
    if "sharpe" in keywords or "threshold" in keywords:
        result = {
            "category": "metric_deviation",
            "root_cause": "spec_issue",
            "recommended": "/dgsf_research → /dgsf_spec_propose"
        }
    elif "interface" in keywords or "contract" in keywords:
        result = {
            "category": "design_issue", 
            "root_cause": "spec_issue",
            "recommended": "/dgsf_research → /dgsf_spec_propose"
        }
    elif "typeerror" in keywords or "attributeerror" in keywords:
        result = {
            "category": "runtime_error",
            "root_cause": "code_bug",
            "recommended": "/dgsf_diagnose"
        }
    else:
        result = {
            "category": "unknown",
            "root_cause": "needs_investigation",
            "recommended": "Manual investigation"
        }
    
    print(f"\n[TRIAGE RESULT]")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    return result


# ============================================================================
# VS Code Copilot 交互指南
# ============================================================================

COPILOT_INTERACTION_GUIDE = """
## VS Code + Copilot 交互验证指南

### 准备工作

1. 确保已安装 Python 扩展和 GitHub Copilot
2. 激活 Python 环境: `.venv/Scripts/activate`
3. 确保 MCP Server 可导入: `python -c "from kernel.mcp_server import MCPServer"`

### 测试 Skill 调用

在 Copilot Chat 中输入以下命令：

#### 测试 1: Spec Triage
```
/dgsf_spec_triage 
问题：实验 t05 的 OOS Sharpe = 0.8，低于阈值 1.5
来源：experiment
```

预期输出：
- Triage ID: TRI-YYYY-MM-DD-XXXXXX
- 分类: metric_deviation
- 根因: spec_issue
- 推荐: /dgsf_research

#### 测试 2: Spec Read
```
读取 projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml 的 validation 部分
```

预期输出：
- 显示当前 validation 配置
- 标注 Layer: L2
- 标注需要 Project Lead 审批

#### 测试 3: Spec Propose
```
/dgsf_spec_propose
Spec: projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml
类型: modify
理由: 将 min_sharpe_threshold 从 1.0 提高到 1.5
```

预期输出：
- 生成 SCP-YYYY-MM-DD-NNN 提案
- 创建 decisions/SCP-*.yaml 文件
- 显示需要 Project Lead 审批

### 验证 Hooks

运行测试检查 hooks 是否正确配置：

```powershell
# 检查 hooks 存在
ls hooks/pre-spec-change
ls hooks/post-spec-change

# 测试 pre-hook（干运行）
sh hooks/pre-spec-change "projects/dgsf/specs/TEST.yaml" "modify" "test-approval"
```

### 运行 E2E 测试

```powershell
pytest projects/dgsf/tests/test_spec_evolution_e2e.py -v
```

### 完整工作流验证

1. 触发问题: 在 Copilot Chat 中描述一个实验失败
2. 观察 Triage: Copilot 应该调用 spec_triage
3. 执行 Research: 输入 /dgsf_research
4. 生成 Proposal: 输入 /dgsf_spec_propose
5. 模拟审批: 创建 decisions/APPROVED.yaml
6. 提交变更: 输入 /dgsf_spec_commit
7. 验证: 检查 lineage 和 audit 记录
"""


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Spec Evolution Workflow - VS Code + Copilot 验证场景")
    print("="*60)
    print(f"时间: {datetime.now().isoformat()}")
    
    # 打印所有场景
    print_scenario(1, SCENARIO_1_PROBLEM)
    simulate_triage_call("OOS Sharpe = 0.8 低于阈值 1.5", "experiment")
    
    print_scenario(2, SCENARIO_2_PROBLEM)
    simulate_triage_call("接口不兼容 PanelTree 输出 features 但 SDF 期望 X", "review")
    
    print_scenario(3, SCENARIO_3_PROBLEM)
    simulate_triage_call("ValidationError max_date_range 限制过严", "test")
    
    print_scenario(4, SCENARIO_4_PROBLEM)
    simulate_triage_call("TypeError unsupported operand NoneType + float", "test")
    
    # 打印交互指南
    print("\n")
    print(COPILOT_INTERACTION_GUIDE)
