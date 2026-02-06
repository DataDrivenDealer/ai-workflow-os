#!/usr/bin/env python3
"""
Subagent Runner — AI Workflow OS
================================

执行 Subagent 任务并生成结构化输出。

用法:
    python kernel/subagent_runner.py <subagent_id> --question "..." [options]

示例:
    # Repo & Specs 检索
    python kernel/subagent_runner.py repo_specs_retrieval \\
        --question "SDF_SPEC v3.1 中定义了哪些特征？" \\
        --scope "specs/"

    # 外部研究
    python kernel/subagent_runner.py external_research \\
        --question "purged walk-forward CV 的最佳实践" \\
        --context "量化策略回测"

    # 量化风险审查
    python kernel/subagent_runner.py quant_risk_review \\
        --files "projects/dgsf/repo/src/dgsf/backtest/engine.py" \\
        --review-type "full"

输出:
    docs/subagents/runs/<timestamp>_<subagent_id>/
    ├── SUMMARY.md       # 主 Agent 消费的简短摘要
    ├── EVIDENCE.md      # 详细证据（路径、行号、引用）
    ├── CHECKLIST.md     # 仅 quant_risk_review
    └── metadata.yaml    # 运行元数据
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 确保可以导入 kernel 模块
KERNEL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(KERNEL_ROOT))

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


# =============================================================================
# CONSTANTS
# =============================================================================

SUBAGENT_REGISTRY_PATH = KERNEL_ROOT / "configs" / "subagent_registry.yaml"
OUTPUT_BASE_DIR = KERNEL_ROOT / "docs" / "subagents" / "runs"


# =============================================================================
# REGISTRY LOADER
# =============================================================================

def load_registry() -> dict:
    """加载 Subagent Registry 配置。"""
    if not SUBAGENT_REGISTRY_PATH.exists():
        print(f"ERROR: Registry not found: {SUBAGENT_REGISTRY_PATH}")
        sys.exit(1)
    
    with open(SUBAGENT_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_subagent_config(registry: dict, subagent_id: str) -> Optional[dict]:
    """获取指定 Subagent 的配置。"""
    subagents = registry.get("subagents", {})
    return subagents.get(subagent_id)


# =============================================================================
# SUBAGENT IMPLEMENTATIONS
# =============================================================================

class SubagentBase:
    """Subagent 基类。"""
    
    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.findings = []
        self.evidence = []
    
    def run(self, **kwargs) -> dict:
        """执行 Subagent 逻辑。由子类实现。"""
        raise NotImplementedError
    
    def write_summary(self, content: str):
        """写入 SUMMARY.md。"""
        (self.output_dir / "SUMMARY.md").write_text(content, encoding="utf-8")
    
    def write_evidence(self, content: str):
        """写入 EVIDENCE.md。"""
        (self.output_dir / "EVIDENCE.md").write_text(content, encoding="utf-8")
    
    def write_metadata(self, metadata: dict):
        """写入 metadata.yaml。"""
        with open(self.output_dir / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False)


class RepoSpecsRetrievalAgent(SubagentBase):
    """本地仓库与规范检索 Subagent。"""
    
    def run(self, question: str, scope: str = ".", 
            file_patterns: Optional[list] = None,
            keywords: Optional[list] = None, **kwargs) -> dict:
        """执行本地搜索。"""
        
        print(f"[RepoSpecsRetrieval] Question: {question}")
        print(f"[RepoSpecsRetrieval] Scope: {scope}")
        
        # 从问题中提取关键词
        if keywords is None:
            keywords = self._extract_keywords(question)
        
        # 执行搜索
        results = []
        for keyword in keywords:
            matches = self._ripgrep_search(keyword, scope, file_patterns)
            results.extend(matches)
        
        # 去重
        unique_results = self._deduplicate(results)
        
        # 生成输出
        summary = self._generate_summary(question, unique_results)
        evidence = self._generate_evidence(unique_results)
        
        self.write_summary(summary)
        self.write_evidence(evidence)
        
        return {
            "status": "success",
            "findings_count": len(unique_results),
            "output_dir": str(self.output_dir)
        }
    
    def _extract_keywords(self, question: str) -> list:
        """从问题中提取搜索关键词。"""
        # 简单实现：提取可能的标识符
        import re
        # 匹配大写标识符、版本号等
        patterns = [
            r'[A-Z][A-Z_]+[A-Z0-9]*',  # SDF_SPEC, DGSF
            r'v\d+\.\d+',              # v3.1
            r'[a-z_]+_[a-z_]+',        # snake_case 标识符
        ]
        keywords = []
        for pattern in patterns:
            keywords.extend(re.findall(pattern, question))
        
        # 如果没有找到，使用整个问题的关键部分
        if not keywords:
            # 提取中文后的关键词或英文单词
            words = question.split()
            keywords = [w for w in words if len(w) > 3][:3]
        
        return list(set(keywords))[:5]  # 最多 5 个关键词
    
    def _ripgrep_search(self, keyword: str, scope: str, 
                        file_patterns: Optional[list] = None) -> list:
        """使用 ripgrep 搜索。"""
        results = []
        
        # 构建 rg 命令
        cmd = ["rg", "-n", "-i", "--json", keyword]
        
        if file_patterns:
            for pattern in file_patterns:
                cmd.extend(["-g", pattern])
        
        search_path = KERNEL_ROOT / scope
        if not search_path.exists():
            search_path = KERNEL_ROOT
        
        cmd.append(str(search_path))
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            
            # 解析 JSON 输出
            import json
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "match":
                        data = entry.get("data", {})
                        results.append({
                            "file": data.get("path", {}).get("text", ""),
                            "line": data.get("line_number", 0),
                            "content": data.get("lines", {}).get("text", "").strip(),
                            "keyword": keyword
                        })
                except json.JSONDecodeError:
                    continue
        except subprocess.TimeoutExpired:
            print(f"[WARN] Search timeout for keyword: {keyword}")
        except FileNotFoundError:
            # ripgrep 不可用，使用 Python 实现
            results = self._python_search(keyword, search_path, file_patterns)
        
        return results[:50]  # 限制结果数量
    
    def _python_search(self, keyword: str, search_path: Path, 
                       file_patterns: Optional[list] = None) -> list:
        """Python 实现的搜索（ripgrep 不可用时的后备方案）。"""
        results = []
        
        patterns = file_patterns or ["*.yaml", "*.md", "*.py"]
        
        for pattern in patterns:
            for file_path in search_path.rglob(pattern):
                # 跳过 .git、__pycache__ 等目录
                if any(p.startswith(".") or p == "__pycache__" 
                       for p in file_path.parts):
                    continue
                
                try:
                    content = file_path.read_text(encoding="utf-8")
                    for i, line in enumerate(content.split("\n"), 1):
                        if keyword.lower() in line.lower():
                            results.append({
                                "file": str(file_path.relative_to(KERNEL_ROOT)),
                                "line": i,
                                "content": line.strip()[:200],
                                "keyword": keyword
                            })
                except Exception:
                    continue
        
        return results[:50]
    
    def _deduplicate(self, results: list) -> list:
        """去重。"""
        seen = set()
        unique = []
        for r in results:
            key = (r["file"], r["line"])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
    
    def _generate_summary(self, question: str, results: list) -> str:
        """生成摘要。"""
        # 按文件分组
        files = {}
        for r in results:
            files.setdefault(r["file"], []).append(r)
        
        summary = f"""# Subagent Summary: Repo & Specs Retrieval

**Question**: {question}

**Confidence**: {"high" if len(results) > 5 else "medium" if results else "low"}

## Key Findings

"""
        if not results:
            summary += "- No matches found for the given query.\n"
        else:
            summary += f"- Found **{len(results)}** matches across **{len(files)}** files.\n"
            
            # 列出主要文件
            top_files = sorted(files.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            for file_path, matches in top_files:
                summary += f"- `{file_path}`: {len(matches)} matches\n"
        
        summary += f"""
## Answer

Based on the search results, {"relevant content was found in the files listed above" if results else "no direct matches were found"}.

---
*Generated by RepoSpecsRetrievalAgent at {datetime.now().isoformat()}*
"""
        return summary
    
    def _generate_evidence(self, results: list) -> str:
        """生成证据文档。"""
        evidence = """# Evidence: Repo & Specs Retrieval

## File References

"""
        if not results:
            evidence += "*No evidence collected.*\n"
            return evidence
        
        # 按文件分组
        files = {}
        for r in results:
            files.setdefault(r["file"], []).append(r)
        
        for file_path, matches in sorted(files.items()):
            evidence += f"### `{file_path}`\n\n"
            for m in matches[:10]:  # 每个文件最多 10 条
                evidence += f"**Line {m['line']}** (keyword: `{m['keyword']}`)\n"
                evidence += f"```\n{m['content'][:300]}\n```\n\n"
        
        return evidence


class ExternalResearchAgent(SubagentBase):
    """外部网络研究 Subagent。"""
    
    def run(self, research_question: str, context: str = "",
            source_types: Optional[list] = None, **kwargs) -> dict:
        """执行外部研究。"""
        
        print(f"[ExternalResearch] Question: {research_question}")
        print(f"[ExternalResearch] Context: {context}")
        
        # 注意：这是一个占位实现
        # 实际使用时，需要集成 web_search API
        
        summary = f"""# Subagent Summary: External Research

**Research Question**: {research_question}

**Context**: {context}

**Confidence**: low

## Recommendations

⚠️ **Note**: This is a placeholder implementation.

To perform actual web research:
1. Configure a web search API (e.g., Bing, Google, or academic APIs)
2. Set the `WEB_SEARCH_API_KEY` environment variable
3. Re-run this subagent

## Manual Research Pointers

Consider searching for:
- Academic papers on arXiv, SSRN
- Documentation on official library sites
- Blog posts from reputable quant researchers

---
*Generated by ExternalResearchAgent at {datetime.now().isoformat()}*
"""
        
        evidence = """# Evidence: External Research

*No external sources fetched in this placeholder run.*

## Suggested Sources

For the research question, consider:
1. arXiv.org (cs.LG, q-fin.CP)
2. SSRN (quantitative finance)
3. GitHub repositories (reference implementations)
4. Official documentation (scikit-learn, statsmodels)
"""
        
        self.write_summary(summary)
        self.write_evidence(evidence)
        
        return {
            "status": "placeholder",
            "note": "External research requires web API configuration",
            "output_dir": str(self.output_dir)
        }


class QuantRiskReviewAgent(SubagentBase):
    """量化风险审查 Subagent。"""
    
    # 常见风险模式
    RISK_PATTERNS = {
        "lookahead_bias": [
            (r"\.shift\(-", "Negative shift may cause lookahead"),
            (r"future", "Reference to 'future' data"),
            (r"\.iloc\[-1\]", "Accessing last row without time context"),
        ],
        "data_leakage": [
            (r"train_test_split.*shuffle.*True", "Shuffle=True in time series"),
            (r"fit_transform.*test", "fit_transform on test data"),
            (r"\.fit\(.*X\)", "Fitting without purging"),
        ],
        "evaluation_protocol": [
            (r"accuracy", "Using accuracy instead of risk-adjusted metrics"),
            (r"cross_val_score.*cv=\d", "Standard CV instead of walk-forward"),
        ],
        "reproducibility": [
            (r"random_state\s*=\s*None", "No random seed set"),
            (r"np\.random\.", "Direct numpy random without seed"),
        ]
    }
    
    def run(self, target_files: list, review_type: str = "full",
            focus_areas: Optional[list] = None, **kwargs) -> dict:
        """执行风险审查。"""
        
        print(f"[QuantRiskReview] Files: {target_files}")
        print(f"[QuantRiskReview] Type: {review_type}")
        
        if focus_areas is None:
            focus_areas = ["lookahead", "leakage", "protocol", "reproducibility"]
        
        # 收集问题
        all_issues = {
            "lookahead_bias": [],
            "data_leakage": [],
            "evaluation_protocol": [],
            "reproducibility": []
        }
        
        for file_path in target_files:
            full_path = KERNEL_ROOT / file_path
            if not full_path.exists():
                continue
            
            try:
                content = full_path.read_text(encoding="utf-8")
                issues = self._scan_file(file_path, content, focus_areas)
                for category, items in issues.items():
                    all_issues[category].extend(items)
            except Exception as e:
                print(f"[WARN] Could not read {file_path}: {e}")
        
        # 计算风险分数
        risk_score = self._calculate_risk_score(all_issues)
        verdict = "pass" if risk_score < 3 else "warn" if risk_score < 6 else "fail"
        
        # 生成输出
        summary = self._generate_summary(target_files, all_issues, verdict, risk_score)
        evidence = self._generate_evidence(all_issues)
        checklist = self._generate_checklist(all_issues)
        
        self.write_summary(summary)
        self.write_evidence(evidence)
        (self.output_dir / "CHECKLIST.md").write_text(checklist, encoding="utf-8")
        
        return {
            "status": "success",
            "verdict": verdict,
            "risk_score": risk_score,
            "issues_count": sum(len(v) for v in all_issues.values()),
            "output_dir": str(self.output_dir)
        }
    
    def _scan_file(self, file_path: str, content: str, focus_areas: list) -> dict:
        """扫描单个文件。"""
        import re
        issues = {k: [] for k in self.RISK_PATTERNS}
        
        lines = content.split("\n")
        
        for category, patterns in self.RISK_PATTERNS.items():
            # 检查是否在 focus_areas 中
            short_name = category.split("_")[0]
            if short_name not in focus_areas and category not in focus_areas:
                continue
            
            for pattern, description in patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues[category].append({
                            "file": file_path,
                            "line": i,
                            "code": line.strip()[:100],
                            "pattern": pattern,
                            "description": description
                        })
        
        return issues
    
    def _calculate_risk_score(self, issues: dict) -> int:
        """计算风险分数（0-10）。"""
        weights = {
            "lookahead_bias": 3,
            "data_leakage": 3,
            "evaluation_protocol": 2,
            "reproducibility": 1
        }
        
        score = 0
        for category, items in issues.items():
            score += min(len(items), 3) * weights.get(category, 1)
        
        return min(score, 10)
    
    def _generate_summary(self, files: list, issues: dict, 
                          verdict: str, risk_score: int) -> str:
        """生成摘要。"""
        critical = sum(len(issues.get(c, [])) for c in ["lookahead_bias", "data_leakage"])
        warnings = sum(len(issues.get(c, [])) for c in ["evaluation_protocol", "reproducibility"])
        
        verdict_emoji = {"pass": "✅", "warn": "⚠️", "fail": "🔴"}[verdict]
        
        summary = f"""# Subagent Summary: Quant Risk Review

**Verdict**: {verdict_emoji} **{verdict.upper()}**

**Risk Score**: {risk_score}/10

## Overview

| Metric | Value |
|--------|-------|
| Files Reviewed | {len(files)} |
| Critical Issues | {critical} |
| Warnings | {warnings} |

## Critical Issues

"""
        if critical == 0:
            summary += "*No critical issues detected.*\n"
        else:
            for category in ["lookahead_bias", "data_leakage"]:
                for issue in issues.get(category, [])[:3]:
                    summary += f"- **{category}**: {issue['description']} (`{issue['file']}:{issue['line']}`)\n"
        
        summary += """
## Warnings

"""
        if warnings == 0:
            summary += "*No warnings.*\n"
        else:
            for category in ["evaluation_protocol", "reproducibility"]:
                for issue in issues.get(category, [])[:3]:
                    summary += f"- **{category}**: {issue['description']} (`{issue['file']}:{issue['line']}`)\n"
        
        summary += f"""
---
*Generated by QuantRiskReviewAgent at {datetime.now().isoformat()}*
"""
        return summary
    
    def _generate_evidence(self, issues: dict) -> str:
        """生成证据文档。"""
        evidence = """# Evidence: Quant Risk Review

"""
        for category, items in issues.items():
            evidence += f"## {category.replace('_', ' ').title()}\n\n"
            if not items:
                evidence += "*No issues found.*\n\n"
                continue
            
            for item in items[:10]:
                evidence += f"""### `{item['file']}:{item['line']}`

**Pattern**: `{item['pattern']}`  
**Description**: {item['description']}

```
{item['code']}
```

"""
        return evidence
    
    def _generate_checklist(self, issues: dict) -> str:
        """生成检查清单。"""
        checklist = """# Quant Risk Review Checklist

"""
        categories = [
            ("lookahead_bias", "Lookahead Bias"),
            ("data_leakage", "Data Leakage"),
            ("evaluation_protocol", "Evaluation Protocol"),
            ("reproducibility", "Reproducibility"),
        ]
        
        for key, title in categories:
            items = issues.get(key, [])
            status = "pass" if not items else "warn" if len(items) < 3 else "fail"
            emoji = {"pass": "✅", "warn": "⚠️", "fail": "🔴"}[status]
            
            checklist += f"## {emoji} {title}\n\n"
            checklist += f"**Status**: {status}\n\n"
            
            if items:
                checklist += "**Issues**:\n"
                for item in items:
                    checklist += f"- [ ] Fix `{item['file']}:{item['line']}`: {item['description']}\n"
            else:
                checklist += "*No issues found.*\n"
            
            checklist += "\n"
        
        return checklist


class SpecDriftAgent(SubagentBase):
    """Spec 漂移检测 Subagent。"""
    
    # 漂移分类
    DRIFT_CATEGORIES = {
        "SPEC_LAG": "Spec 落后于实现 — 实现有新功能未在 Spec 中记录",
        "CODE_DRIFT": "实现偏离 Spec — 代码与 Spec 定义不一致",
        "MUTUAL_INCONSISTENCY": "Specs 之间存在冲突定义"
    }
    
    def run(self, scope: str = "specs/", compare_to: str = "projects/dgsf/repo/src/",
            spec_files: Optional[list] = None, check_cross_refs: bool = True,
            **kwargs) -> dict:
        """执行 Spec 漂移检测。"""
        
        print(f"[SpecDrift] Spec scope: {scope}")
        print(f"[SpecDrift] Compare to: {compare_to}")
        
        drift_items = []
        
        # 1. 收集 Spec 文件
        spec_path = KERNEL_ROOT / scope
        if spec_files:
            specs = [KERNEL_ROOT / f for f in spec_files if (KERNEL_ROOT / f).exists()]
        elif spec_path.exists():
            specs = list(spec_path.glob("*.yaml")) + list(spec_path.glob("*.md"))
        else:
            specs = []
        
        # 2. 收集实现文件
        impl_path = KERNEL_ROOT / compare_to
        if impl_path.exists():
            impl_files = list(impl_path.rglob("*.py"))
        else:
            impl_files = []
        
        # 3. 分析每个 Spec
        for spec_file in specs:
            spec_drift = self._analyze_spec(spec_file, impl_files)
            drift_items.extend(spec_drift)
        
        # 4. 检查 Spec 之间的交叉引用冲突
        if check_cross_refs:
            cross_ref_issues = self._check_cross_references(specs)
            drift_items.extend(cross_ref_issues)
        
        # 5. 按类别分组
        by_category = {cat: [] for cat in self.DRIFT_CATEGORIES}
        for item in drift_items:
            cat = item.get("category", "CODE_DRIFT")
            if cat in by_category:
                by_category[cat].append(item)
        
        # 6. 生成输出
        summary = self._generate_summary(specs, impl_files, by_category)
        evidence = self._generate_evidence(by_category)
        
        self.write_summary(summary)
        self.write_evidence(evidence)
        
        return {
            "status": "success",
            "drift_count": len(drift_items),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "output_dir": str(self.output_dir)
        }
    
    def _analyze_spec(self, spec_file: Path, impl_files: list) -> list:
        """分析单个 Spec 与实现的一致性。"""
        import re
        drift_items = []
        
        try:
            content = spec_file.read_text(encoding="utf-8")
        except Exception:
            return drift_items
        
        # 提取 Spec 中定义的标识符
        identifiers = self._extract_identifiers(content)
        
        # 在实现文件中查找
        for impl_file in impl_files:
            try:
                impl_content = impl_file.read_text(encoding="utf-8")
            except Exception:
                continue
            
            impl_identifiers = self._extract_identifiers(impl_content)
            
            # 检测 SPEC_LAG: 实现中有但 Spec 中没有
            for impl_id in impl_identifiers:
                if impl_id not in identifiers:
                    # 检查是否是相关文件
                    spec_name = spec_file.stem.lower()
                    impl_name = impl_file.stem.lower()
                    if spec_name in impl_name or impl_name in spec_name:
                        drift_items.append({
                            "category": "SPEC_LAG",
                            "spec_file": str(spec_file.relative_to(KERNEL_ROOT)),
                            "impl_file": str(impl_file.relative_to(KERNEL_ROOT)),
                            "identifier": impl_id,
                            "description": f"实现中存在 `{impl_id}` 但 Spec 中未定义"
                        })
            
            # 检测 CODE_DRIFT: Spec 中有但实现签名不匹配
            # (简化实现：仅检测命名差异)
            for spec_id in identifiers:
                similar = [i for i in impl_identifiers 
                          if self._similar_name(spec_id, i) and spec_id != i]
                for sim in similar:
                    drift_items.append({
                        "category": "CODE_DRIFT",
                        "spec_file": str(spec_file.relative_to(KERNEL_ROOT)),
                        "impl_file": str(impl_file.relative_to(KERNEL_ROOT)),
                        "spec_identifier": spec_id,
                        "impl_identifier": sim,
                        "description": f"Spec 定义 `{spec_id}` 但实现使用 `{sim}`"
                    })
        
        return drift_items[:20]  # 限制每个 Spec 的问题数量
    
    def _extract_identifiers(self, content: str) -> set:
        """从内容中提取标识符。"""
        import re
        identifiers = set()
        
        # Python 函数/类定义
        identifiers.update(re.findall(r'def\s+(\w+)', content))
        identifiers.update(re.findall(r'class\s+(\w+)', content))
        
        # YAML 键
        identifiers.update(re.findall(r'^  (\w+):', content, re.MULTILINE))
        identifiers.update(re.findall(r'^    - name:\s*(\w+)', content, re.MULTILINE))
        
        # 常量定义
        identifiers.update(re.findall(r'^([A-Z][A-Z_]+)\s*=', content, re.MULTILINE))
        
        # 过滤常见词
        common = {'def', 'class', 'self', 'True', 'False', 'None', 'if', 'else', 
                  'for', 'in', 'return', 'import', 'from', 'as', 'with', 'try', 
                  'except', 'finally', 'raise', 'pass', 'break', 'continue'}
        
        return identifiers - common
    
    def _similar_name(self, name1: str, name2: str) -> bool:
        """判断两个名称是否相似。"""
        # 简单实现：去掉下划线后比较
        n1 = name1.lower().replace("_", "")
        n2 = name2.lower().replace("_", "")
        
        # 长度差距太大不相似
        if abs(len(n1) - len(n2)) > 5:
            return False
        
        # 一个是另一个的子串
        if n1 in n2 or n2 in n1:
            return True
        
        # 共同前缀超过一半
        common_prefix = 0
        for c1, c2 in zip(n1, n2):
            if c1 == c2:
                common_prefix += 1
            else:
                break
        
        return common_prefix >= min(len(n1), len(n2)) * 0.5
    
    def _check_cross_references(self, specs: list) -> list:
        """检查 Spec 之间的交叉引用冲突。"""
        issues = []
        
        # 收集所有 Spec 中的阈值定义
        thresholds = {}  # {name: [(spec_file, value), ...]}
        
        for spec_file in specs:
            try:
                content = spec_file.read_text(encoding="utf-8")
                # 查找阈值定义
                import re
                matches = re.findall(r'(\w+_threshold|\w+_sharpe|\w+_drawdown):\s*(\S+)', 
                                    content, re.IGNORECASE)
                for name, value in matches:
                    thresholds.setdefault(name.lower(), []).append(
                        (str(spec_file.relative_to(KERNEL_ROOT)), value)
                    )
            except Exception:
                continue
        
        # 检测不一致
        for name, definitions in thresholds.items():
            if len(definitions) > 1:
                values = set(d[1] for d in definitions)
                if len(values) > 1:
                    issues.append({
                        "category": "MUTUAL_INCONSISTENCY",
                        "identifier": name,
                        "definitions": definitions,
                        "description": f"`{name}` 在多个 Spec 中定义了不同的值: {values}"
                    })
        
        return issues
    
    def _generate_summary(self, specs: list, impl_files: list, 
                          by_category: dict) -> str:
        """生成摘要。"""
        total = sum(len(v) for v in by_category.values())
        
        summary = f"""# Subagent Summary: Spec Drift Analysis

**Total Drift Items**: {total}

## Specs Analyzed

| Metric | Value |
|--------|-------|
| Spec Files | {len(specs)} |
| Implementation Files | {len(impl_files)} |

## By Category

| Category | Count | Description |
|----------|-------|-------------|
"""
        for cat, desc in self.DRIFT_CATEGORIES.items():
            count = len(by_category.get(cat, []))
            summary += f"| {cat} | {count} | {desc.split(' — ')[0]} |\n"
        
        summary += """
## Recommendations

"""
        for cat, items in by_category.items():
            if not items:
                continue
            item = items[0]
            if cat == "SPEC_LAG":
                summary += f"""1. **[{cat}]** {item.get('spec_file', 'spec')}: 
   {item.get('description', '')}
   → 建议: 更新 Spec 添加新定义

"""
            elif cat == "CODE_DRIFT":
                summary += f"""2. **[{cat}]** {item.get('spec_file', 'spec')} vs {item.get('impl_file', 'impl')}:
   {item.get('description', '')}
   → 建议: 统一命名

"""
            elif cat == "MUTUAL_INCONSISTENCY":
                summary += f"""3. **[{cat}]** {item.get('identifier', '')}:
   {item.get('description', '')}
   → 建议: 调和 Specs 的定义

"""
        
        if total == 0:
            summary += "*No drift detected. Specs and implementation are consistent.*\n"
        
        summary += f"""
---
*Generated by SpecDriftAgent at {datetime.now().isoformat()}*
"""
        return summary
    
    def _generate_evidence(self, by_category: dict) -> str:
        """生成证据文档。"""
        evidence = """# Evidence: Spec Drift Analysis

"""
        for cat, items in by_category.items():
            evidence += f"## {cat}\n\n"
            
            if not items:
                evidence += "*No issues found.*\n\n"
                continue
            
            for i, item in enumerate(items[:10], 1):
                evidence += f"### Issue {i}\n\n"
                evidence += f"**Category**: {cat}\n\n"
                
                if "spec_file" in item:
                    evidence += f"**Spec File**: `{item['spec_file']}`\n\n"
                if "impl_file" in item:
                    evidence += f"**Implementation File**: `{item['impl_file']}`\n\n"
                if "identifier" in item:
                    evidence += f"**Identifier**: `{item['identifier']}`\n\n"
                
                evidence += f"**Description**: {item.get('description', 'N/A')}\n\n"
                
                if "definitions" in item:
                    evidence += "**Conflicting Definitions**:\n"
                    for spec, value in item["definitions"]:
                        evidence += f"- `{spec}`: {value}\n"
                    evidence += "\n"
                
                evidence += "---\n\n"
        
        return evidence


# =============================================================================
# MAIN RUNNER
# =============================================================================

SUBAGENT_CLASSES = {
    "repo_specs_retrieval": RepoSpecsRetrievalAgent,
    "external_research": ExternalResearchAgent,
    "quant_risk_review": QuantRiskReviewAgent,
    "spec_drift": SpecDriftAgent,
}


def run_subagent(subagent_id: str, args: argparse.Namespace) -> dict:
    """运行指定的 Subagent。"""
    
    # 加载 registry
    registry = load_registry()
    config = get_subagent_config(registry, subagent_id)
    
    if config is None:
        print(f"ERROR: Unknown subagent: {subagent_id}")
        print(f"Available: {list(registry.get('subagents', {}).keys())}")
        sys.exit(1)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE_DIR / f"{timestamp}_{subagent_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取 subagent 类
    agent_class = SUBAGENT_CLASSES.get(subagent_id)
    if agent_class is None:
        print(f"ERROR: No implementation for subagent: {subagent_id}")
        sys.exit(1)
    
    # 实例化并运行
    agent = agent_class(config, output_dir)
    
    # 准备参数
    kwargs = {}
    if hasattr(args, "question") and args.question:
        kwargs["question"] = args.question
        kwargs["research_question"] = args.question  # 兼容 external_research
    if hasattr(args, "scope") and args.scope:
        kwargs["scope"] = args.scope
    if hasattr(args, "context") and args.context:
        kwargs["context"] = args.context
    if hasattr(args, "files") and args.files:
        kwargs["target_files"] = args.files
    if hasattr(args, "review_type") and args.review_type:
        kwargs["review_type"] = args.review_type
    if hasattr(args, "keywords") and args.keywords:
        kwargs["keywords"] = args.keywords
    if hasattr(args, "focus_areas") and args.focus_areas:
        kwargs["focus_areas"] = args.focus_areas
    if hasattr(args, "compare_to") and args.compare_to:
        kwargs["compare_to"] = args.compare_to
    if hasattr(args, "spec_files") and args.spec_files:
        kwargs["spec_files"] = args.spec_files
    
    # 运行
    print(f"\n{'='*60}")
    print(f"Running Subagent: {subagent_id}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}\n")
    
    result = agent.run(**kwargs)
    
    # 写入元数据
    metadata = {
        "subagent_id": subagent_id,
        "timestamp": datetime.now().isoformat(),
        "input": kwargs,
        "result": result
    }
    agent.write_metadata(metadata)
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"Subagent Complete: {subagent_id}")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # 显示摘要
    summary_path = output_dir / "SUMMARY.md"
    if summary_path.exists():
        print("--- SUMMARY.md ---")
        print(summary_path.read_text(encoding="utf-8")[:2000])
        if len(summary_path.read_text(encoding="utf-8")) > 2000:
            print("\n... (truncated) ...")
    
    return result


def main():
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="Subagent Runner — Execute subagent tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repo & Specs Retrieval
  python kernel/subagent_runner.py repo_specs_retrieval \\
      --question "What features are defined in SDF_SPEC v3.1?"

  # External Research
  python kernel/subagent_runner.py external_research \\
      --question "Best practices for purged walk-forward CV"

  # Quant Risk Review
  python kernel/subagent_runner.py quant_risk_review \\
      --files projects/dgsf/repo/src/dgsf/backtest/engine.py
        """
    )
    
    parser.add_argument(
        "subagent_id",
        nargs="?",  # 可选位置参数
        choices=list(SUBAGENT_CLASSES.keys()),
        help="Subagent to run"
    )
    
    parser.add_argument(
        "--question", "-q",
        help="Question to answer (for repo_specs_retrieval, external_research)"
    )
    
    parser.add_argument(
        "--scope", "-s",
        default=".",
        help="Search scope relative to workspace root (default: .)"
    )
    
    parser.add_argument(
        "--context", "-c",
        help="Context for external research"
    )
    
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="Files to review (for quant_risk_review)"
    )
    
    parser.add_argument(
        "--review-type", "-t",
        choices=["full", "incremental", "focused"],
        default="full",
        help="Review type (default: full)"
    )
    
    parser.add_argument(
        "--keywords", "-k",
        nargs="+",
        help="Keywords for search"
    )
    
    parser.add_argument(
        "--focus-areas",
        nargs="+",
        choices=["lookahead", "leakage", "protocol", "reproducibility"],
        help="Focus areas for risk review"
    )
    
    parser.add_argument(
        "--compare-to",
        default="projects/dgsf/repo/src/",
        help="Implementation path to compare specs against (for spec_drift)"
    )
    
    parser.add_argument(
        "--spec-files",
        nargs="+",
        help="Specific spec files to analyze (for spec_drift)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available subagents"
    )
    
    args = parser.parse_args()
    
    if args.list:
        registry = load_registry()
        print("Available Subagents:")
        for sid, config in registry.get("subagents", {}).items():
            print(f"  - {sid}: {config.get('purpose', '').split('.')[0]}")
        return
    
    # 如果没有提供 subagent_id 且没有 --list，显示帮助
    if not args.subagent_id:
        parser.print_help()
        return
    
    # 验证必需参数
    if args.subagent_id == "repo_specs_retrieval" and not args.question:
        parser.error("repo_specs_retrieval requires --question")
    if args.subagent_id == "external_research" and not args.question:
        parser.error("external_research requires --question")
    if args.subagent_id == "quant_risk_review" and not args.files:
        parser.error("quant_risk_review requires --files")
    # spec_drift 不需要必需参数，有默认值
    
    run_subagent(args.subagent_id, args)


if __name__ == "__main__":
    main()
