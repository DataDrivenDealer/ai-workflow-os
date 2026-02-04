# AI Workflow OS — Capability Matrix

> **Version**: 1.0.0 | **Generated**: 2026-02-04 | **Validation**: All tests passing (237 kernel + 17 E2E)

This document records the **verified runtime capabilities** of the AI Workflow OS system,
specifically focused on what GitHub Copilot can actually leverage at runtime.

## Runtime Constraint Understanding

GitHub Copilot operates with three primary integration points:

| Integration Point | Implementation | Copilot Access | Status |
|-------------------|----------------|----------------|--------|
| **Skills** (`.prompt.md` files) | `.github/prompts/dgsf_*.prompt.md` | ✅ Direct invocation via `/skill` | **VERIFIED** |
| **File Read/Write** | VS Code native tools | ✅ Full access | **VERIFIED** |
| **Terminal Commands** | VS Code native tools | ✅ Full access | **VERIFIED** |
| **MCP Server** | `kernel/mcp_server.py` | ❌ Not directly integrated | External tools only |

## Verified Capabilities

### Category 1: Skills (20 prompts)

All skills are verified to exist and follow consistent structure:

| Skill ID | Purpose | Prompt File | Tested |
|----------|---------|-------------|--------|
| `/dgsf_research` | Explore before planning | [dgsf_research.prompt.md](.github/prompts/dgsf_research.prompt.md) | ✅ |
| `/dgsf_plan` | Define steps + success criteria | [dgsf_plan.prompt.md](.github/prompts/dgsf_plan.prompt.md) | ✅ |
| `/dgsf_execute` | Implement + test | [dgsf_execute.prompt.md](.github/prompts/dgsf_execute.prompt.md) | ✅ |
| `/dgsf_verify` | Validate claims against thresholds | [dgsf_verify.prompt.md](.github/prompts/dgsf_verify.prompt.md) | ✅ |
| `/dgsf_diagnose` | Find root cause of failure | [dgsf_diagnose.prompt.md](.github/prompts/dgsf_diagnose.prompt.md) | ✅ |
| `/dgsf_abort` | Exit infeasible path | [dgsf_abort.prompt.md](.github/prompts/dgsf_abort.prompt.md) | ✅ |
| `/dgsf_decision_log` | Record key decisions | [dgsf_decision_log.prompt.md](.github/prompts/dgsf_decision_log.prompt.md) | ✅ |
| `/dgsf_state_update` | Track progress | [dgsf_state_update.prompt.md](.github/prompts/dgsf_state_update.prompt.md) | ✅ |
| `/dgsf_research_summary` | Synthesize findings | [dgsf_research_summary.prompt.md](.github/prompts/dgsf_research_summary.prompt.md) | ✅ |
| `/dgsf_repo_scan` | Understand codebase state | [dgsf_repo_scan.prompt.md](.github/prompts/dgsf_repo_scan.prompt.md) | ✅ |
| `/dgsf_git_ops` | Git commit/branch/merge | [dgsf_git_ops.prompt.md](.github/prompts/dgsf_git_ops.prompt.md) | ✅ |
| `/dgsf_spec_triage` | Classify spec-related issues | [dgsf_spec_triage.prompt.md](.github/prompts/dgsf_spec_triage.prompt.md) | ✅ |
| `/dgsf_spec_propose` | Propose spec changes | [dgsf_spec_propose.prompt.md](.github/prompts/dgsf_spec_propose.prompt.md) | ✅ |
| `/dgsf_spec_commit` | Apply approved spec changes | [dgsf_spec_commit.prompt.md](.github/prompts/dgsf_spec_commit.prompt.md) | ✅ |
| `/dgsf_knowledge_sync` | Query Quant Knowledge Base | [dgsf_knowledge_sync.prompt.md](.github/prompts/dgsf_knowledge_sync.prompt.md) | ✅ |
| `/dgsf_practice_check` | Verify code against CPR | [dgsf_practice_check.prompt.md](.github/prompts/dgsf_practice_check.prompt.md) | ✅ |
| `/dgsf_protocol_design` | Design experiment using RPA | [dgsf_protocol_design.prompt.md](.github/prompts/dgsf_protocol_design.prompt.md) | ✅ |
| `/dgsf_debt_review` | Prioritize technical/research debt | [dgsf_debt_review.prompt.md](.github/prompts/dgsf_debt_review.prompt.md) | ✅ |
| `/dgsf_memory_query` | Query institutional memory | [dgsf_memory_query.prompt.md](.github/prompts/dgsf_memory_query.prompt.md) | ⚠️ |
| `/dgsf_threshold_resolve` | Get context-aware thresholds | [dgsf_threshold_resolve.prompt.md](.github/prompts/dgsf_threshold_resolve.prompt.md) | ✅ |

> ⚠️ `/dgsf_memory_query` has a prompt file but no backend storage. It will work as a prompt template but cannot persist or query historical data.

### Category 2: Knowledge Base Configs

These YAML files provide structured knowledge that Copilot can read via `read_file`:

| Config | Path | Lines | Content |
|--------|------|-------|---------|
| Quant Knowledge Base (QKB) | `configs/quant_knowledge_base.yaml` | 368 | Factor research consensus, methodology standards |
| Code Practice Registry (CPR) | `configs/code_practice_registry.yaml` | 655 | Anti-patterns, correct patterns, enforcement |
| Research Protocol Algebra (RPA) | `configs/research_protocol_algebra.yaml` | 521 | Composable experiment primitives |
| Adaptive Threshold Engine (ATE) | `configs/adaptive_threshold_engine.yaml` | - | Regime-aware thresholds |
| Strategic Debt Ledger (SDL) | `configs/strategic_debt_ledger.yaml` | - | Debt tracking schema |

### Category 3: MCP Server Tools

These tools are implemented in `kernel/mcp_server.py` and tested. They can be invoked via CLI or external AI tools, **not directly by GitHub Copilot**:

| Tool | Test Status | Implementation Lines |
|------|-------------|---------------------|
| `agent_register` | ✅ Tested | L786-814 |
| `session_create` | ✅ Tested | L830-848 |
| `session_validate` | ✅ Tested | L850-865 |
| `session_terminate` | ✅ Tested | L867-874 |
| `task_list` | ✅ Tested | L878-908 |
| `task_get` | ✅ Tested | L910-940 |
| `task_start` | ✅ Tested | L942-1012 |
| `task_finish` | ✅ Tested | L1014-1040 |
| `governance_check` | ✅ Tested | L1044-1066 |
| `artifact_read` | ✅ Tested | L1070-1100 |
| `artifact_list` | ✅ Tested | L1102-1130 |
| `spec_list` | ✅ Tested | L1134-1180 |
| `spec_read` | ✅ Tested | L1185-1240 |
| `spec_propose` | ✅ Tested | L1242-1333 |
| `spec_commit` | ✅ Tested | L1335-1438 |
| `spec_triage` | ✅ Tested | L1440-1560 |
| `review_*` (7 tools) | ✅ Tested | L1600+ |

### Category 4: Git Hooks

| Hook | Path | Trigger | Status |
|------|------|---------|--------|
| `pre-commit` | `hooks/pre-commit` | Git commit | ✅ Exists |
| `pre-push` | `hooks/pre-push` | Git push | ✅ Exists |
| `pre-spec-change` | `hooks/pre-spec-change` | Manual / Skill | ✅ Exists (190 lines) |
| `post-spec-change` | `hooks/post-spec-change` | Manual / Skill | ✅ Exists |

## Removed/Downgraded Capabilities

The following were removed or downgraded in the v8.0 convergence:

| Capability | Previous State | New State | Reason |
|-----------|----------------|-----------|--------|
| "AUTOMATIC TRIGGERS" | Implied programmatic execution | "RECOMMENDED PATTERNS" | Copilot doesn't auto-execute |
| MCP direct invocation | "可在对话中直接调用" | External tools only | Copilot doesn't integrate MCP |
| IMG auto-record | "做出重要决策 → IMG 记录" | Manual file creation | No IMG backend |
| SDL auto-register | "发现技术债务 → SDL 登记" | Manual file creation | SDL is config only |
| REL rule engine | Formal verification implied | Natural language rules | REL not implemented |

## Test Evidence

```
$ pytest kernel/tests/ -v --tb=short
================================================= 237 passed in 40.72s =================================================

$ pytest projects/dgsf/tests/test_spec_evolution_e2e.py -v
================================================= 17 passed in 0.55s ==================================================
```

## Rollback Procedure

If issues arise, revert to pre-convergence state:

```bash
git log --oneline -5  # Find the commit before convergence
git revert <commit-hash>
pytest kernel/tests/ -v  # Verify tests still pass
```

---

*Generated by architecture convergence process, 2026-02-04*
