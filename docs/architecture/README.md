# Architecture Design Documents

This directory contains **design documents** that describe the conceptual architecture of the Copilot Runtime OS.

> ⚠️ **Important**: These files are **not loaded at runtime** by GitHub Copilot.
> They serve as reference documentation for understanding the system design.

## Runtime vs Design Documents

| Category | Files | Loaded by Copilot? |
|----------|-------|-------------------|
| **Runtime** | `.github/copilot-instructions.md` | ✅ Yes |
| **Runtime** | `.github/prompts/*.prompt.md` | ✅ Yes (as skills) |
| **Design** | `docs/architecture/*.yaml` | ❌ No |
| **Helper** | `kernel/*.py` | ❌ No (manual invocation) |

## Document Index

| File | Description |
|------|-------------|
| [meta_model.yaml](meta_model.yaml) | Four-layer architecture conceptual model |
| [project_interface.yaml](project_interface.yaml) | Abstract interface contract between Kernel and Projects |
| [evolution_policy.yaml](evolution_policy.yaml) | Evolution signal aggregation and review policies |
| [skill_alignment.yaml](skill_alignment.yaml) | Skill-to-workflow alignment mapping |

## Why Separate?

These design documents were moved from `configs/` to clarify:

1. **No runtime loading**: Copilot doesn't automatically parse YAML configs
2. **Reduce confusion**: Clearly separate "what influences behavior" from "what documents intent"
3. **Maintainability**: Design can evolve independently of runtime behavior

## Related

- [copilot-instructions.md](../../.github/copilot-instructions.md) — Actual runtime behavior rules
- [EVOLUTION_LOG.md](../../.github/EVOLUTION_LOG.md) — Change history
