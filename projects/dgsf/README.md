# DGSF Project Reference

**Project**: Dynamic Generative SDF Forest (Asset Pricing Research Framework)  
**Remote**: https://github.com/DataDrivenDealer/DGSF.git  
**Integration**: Git submodule at `projects/dgsf/repo`  
**Status**: Active Development (Stage 4: Research Continuation)  
**Last Updated**: 2026-02-02

---

## 📁 Directory Structure

```
projects/dgsf/
├── repo/                    # ✅ ACTIVE: Git submodule - primary development location
│   ├── src/dgsf/           # Source code (paneltree, sdf, ea, rolling, dataeng)
│   ├── tests/              # Test suite (pytest)
│   ├── configs/            # Experiment configurations
│   ├── scripts/            # Execution scripts (runners, baselines)
│   └── docs/               # DGSF-specific documentation
├── adapter/                # DGSF ↔ AI Workflow OS integration layer
│   ├── dgsf_adapter.py     # Main adapter
│   ├── spec_mapper.py      # Specification resolution
│   ├── task_hooks.py       # Task lifecycle hooks
│   └── data_loader.py      # Data loading utilities
├── specs/                  # L2 Project specifications
│   ├── PROJECT_DGSF.yaml   # Master project spec (v2.1.0)
│   └── SDF_INTERFACE_CONTRACT.yaml
├── docs/                   # AI Workflow OS-managed documentation
│   ├── ARCH_REUSE_ASSESSMENT.md
│   ├── DATA_ASSET_INVENTORY.md
│   ├── SPEC_MAPPING_PLAN.md
│   └── ...
├── data/                   # Data assets (managed by OS)
│   ├── raw/
│   ├── processed/
│   ├── snapshots/
│   └── checksums.yaml
├── legacy/                 # ⚠️ ARCHIVED: Historical assets (DO NOT MODIFY)
│   └── README.md           # Archive warning
└── README.md               # This file
```

---

## � Daily Workflow Checklist

**Purpose**: 标准化 DGSF 日常开发流程，减少认知负载  
**Duration**: ~5 minutes  
**Frequency**: 每次开始工作前

### Morning Routine（开始工作前）

```powershell
# Step 1: Navigate to DGSF repo
cd "E:\AI Tools\AI Workflow OS\projects\dgsf\repo"

# Step 2: Quick health check (使用快速验证脚本)
& "..\..\scripts\dgsf_quick_check.ps1"

# Step 3: Sync with remote (pull latest changes)
git pull origin master

# Step 4: Verify test environment
pytest tests/ --collect-only -q | Select-Object -Last 3
# Expected: "XXX tests in YY.ZZs"

# Step 5: (Optional) Check experiment logs if running long tasks
Get-ChildItem experiments/ -Recurse -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 3
```

### Development Cycle（开发迭代中）

```powershell
# 1. Make code changes in src/dgsf/

# 2. Run unit tests (fast feedback)
pytest tests/test_sdf.py -v -x  # Stop on first failure

# 3. (Optional) Run integration tests
pytest tests/sdf/ -v --tb=short

# 4. Commit incrementally (small, frequent commits)
git add src/dgsf/sdf/my_feature.py
git commit -m "feat(sdf): implement feature X (WIP)"
```

### Evening Routine（结束工作前）

```powershell
# Step 1: Check uncommitted changes
git status

# Step 2: Commit or stash work
git add .
git commit -m "chore: end-of-day checkpoint"
# OR: git stash save "WIP: feature description"

# Step 3: Push to remote (backup)
git push origin <your-branch>

# Step 4: (Optional) Check pending experiments
Get-ChildItem experiments/ -Recurse -Filter "results.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 3
```

### Troubleshooting Quick Reference

| Issue | Command | Notes |
|-------|---------|-------|
| Tests not collecting | `pytest --cache-clear` | Clear pytest cache |
| Import errors | `python -c "import dgsf; print(dgsf.__file__)"` | Verify package installed |
| Submodule out of sync | `git submodule update --init --recursive` | From AI Workflow OS root |
| Merge conflicts | `git status ; git diff` | Check conflict markers |

---

## 🚀 Quick Start for DGSF Researchers

```powershell
# 1. Navigate to the active development directory
cd projects/dgsf/repo

# 2. Check DGSF repo status
git status
git log --oneline -5

# 3. Create a feature branch (if needed)
git checkout -b feature/my-experiment

# 4. Run tests to verify environment
pytest tests/ -v

# 5. Make your changes in src/dgsf/

# 6. Run specific test module
pytest tests/test_sdf.py -v

# 7. Commit changes (in DGSF repo)
git add src/dgsf/sdf/my_model.py
git commit -m "feat(sdf): add new pricing model"

# 8. Push to DGSF remote
git push origin feature/my-experiment
```

### Where to Work?

| Task | Correct Location | Notes |
|------|-----------------|-------|
| 📝 Implement new SDF model | `repo/src/dgsf/sdf/` | Active codebase |
| 🧪 Add unit tests | `repo/tests/` | Use pytest |
| ⚙️ Create experiment config | `repo/configs/` | YAML format |
| 📊 Run baseline | `repo/scripts/` | Use baseline runners |
| 📄 Update DGSF spec | `specs/PROJECT_DGSF.yaml` | Coordinate with AI Workflow OS |
| 🔌 Modify adapter | `adapter/` | OS ↔ DGSF integration |
| 📦 Access data | `data/` | Managed by AI Workflow OS |
| ⚠️ View legacy code | `legacy/` (read-only) | Reference only, DO NOT MODIFY |

---

## 🧪 Testing

### Run DGSF Tests

```powershell
# In projects/dgsf/repo/
pytest tests/ -v                    # All tests
pytest tests/test_sdf.py -v        # Specific module
pytest -k "test_model" -v          # By keyword
pytest --cov=src/dgsf --cov-report=html  # With coverage
```

### Run AI Workflow OS Tests (Kernel)

```powershell
# In AI Workflow OS root
pytest kernel/tests/ -v            # Kernel tests only (excludes DGSF legacy)
```

**Note**: The root `pytest.ini` is configured to exclude `projects/dgsf/legacy/` to prevent 165 collection errors.

---

## 📋 Typical Research Tasks

### Task 1: Reproduce a Baseline (e.g., Baseline A)

```powershell
cd projects/dgsf/repo

# Check baseline runner script
cat scripts/baseline_a_runner.py

# Run baseline A
python scripts/baseline_a_runner.py --config configs/baseline_a.yaml

# Verify results
cat results/baseline_a/report.json
```

### Task 2: Implement a New SDF Variant

```powershell
cd projects/dgsf/repo

# Create new model file
New-Item -Path src/dgsf/sdf/my_sdf_model.py

# Implement your model (inherit from SDFBase)
# Add tests in tests/test_my_sdf_model.py

# Run tests
pytest tests/test_my_sdf_model.py -v

# Update experiment config
cp configs/template_sdf.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your model parameters

# Run experiment
python scripts/run_sdf_experiment.py --config configs/my_experiment.yaml
```

### Task 3: Update Project Specification

```powershell
cd "E:\AI Tools\AI Workflow OS"

# Edit L2 spec (requires OS governance)
code projects/dgsf/specs/PROJECT_DGSF.yaml

# After editing, run spec validation (if available)
python scripts/validate_spec.py projects/dgsf/specs/PROJECT_DGSF.yaml

# Commit spec change
git add projects/dgsf/specs/PROJECT_DGSF.yaml
git commit -m "docs(dgsf): update Stage 4 research tasks"
```

---

## 🔗 Integration with AI Workflow OS

### Adapter Layer

The `projects/dgsf/adapter/` provides integration between DGSF and AI Workflow OS:

- **dgsf_adapter.py**: Main entry point for OS → DGSF calls
- **spec_mapper.py**: Maps legacy DGSF specs to OS L2 specs
- **task_hooks.py**: Lifecycle hooks (task start/finish/audit)
- **data_loader.py**: Unified data loading from `projects/dgsf/data/`

### Governance Integration

DGSF follows AI Workflow OS governance:
- **Authority**: PROJECT_DGSF.yaml (L2 spec, accepted by Project Owner)
- **Gates**: G1 (spec integration), G2 (data migration), G3 (reproducibility)
- **Audit**: All DGSF tasks logged to `ops/audit/dgsf/`

---

## 📚 Key Documentation

### DGSF-Specific (in `repo/`)
- [DGSF Architecture v3.0](repo/docs/specs_v3/DGSF%20Architecture%20v3.0%20_%20Final.md)
- [SDF Layer Spec v3.1](repo/docs/specs_v3/DGSF%20SDF%20Layer%20Specification%20v3.1.md)
- [PanelTree Spec v3.0.2](repo/docs/specs_v3/DGSF%20PanelTree%20Layer%20Specification%20v3.0.2.md)
- [Baseline System Spec v4.3](repo/docs/specs_v3/DGSF%20Baseline%20System%20Specification%20v4.3.md)

### AI Workflow OS Integration
- [EXECUTION_PLAN_DGSF_V1.md](../../docs/plans/EXECUTION_PLAN_DGSF_V1.md) - Current execution plan
- [PROJECT_DGSF.yaml](specs/PROJECT_DGSF.yaml) - L2 project specification
- [SPEC_MAPPING_PLAN.md](docs/SPEC_MAPPING_PLAN.md) - Legacy → OS spec mapping
- [DATA_ASSET_INVENTORY.md](docs/DATA_ASSET_INVENTORY.md) - Data assets catalog

---

## 🚨 Common Pitfalls

### ❌ DO NOT:
1. **Modify files in `projects/dgsf/legacy/`** - This is an archive. Work in `repo/` instead.
2. **Run `pytest .` from AI Workflow OS root** - Use `pytest kernel/tests/` to avoid legacy errors.
3. **Hard-code paths like `e:\DGSF\`** - Use relative paths or `kernel/paths.py` utilities.
4. **Push to DGSF repo without running tests** - Always run `pytest tests/` first.
5. **Edit DGSF specs without OS coordination** - Update `specs/PROJECT_DGSF.yaml` and coordinate with governance.

### ✅ DO:
1. **Always work in `projects/dgsf/repo/`** for active development
2. **Use `pytest tests/` in DGSF repo** for DGSF-specific tests
3. **Update submodule** if behind: `cd repo && git pull origin master`
4. **Check `specs/PROJECT_DGSF.yaml`** for current stage and tasks
5. **Reference `legacy/` for historical context** but never modify

---

## 🔄 Submodule Management

### Update DGSF Repo Submodule

```powershell
# In AI Workflow OS root
cd projects/dgsf/repo
git fetch origin
git merge origin/master
cd ../..
git add projects/dgsf/repo
git commit -m "chore(dgsf): update submodule to latest"
```

### Verify Submodule Status

```powershell
cd projects/dgsf/repo
git status                  # Should be "up to date with origin/master"
git log --oneline -5        # Check recent commits
```

---

## 🎯 Current Status (Stage 4)

**Pipeline**: Stage 4 - Research Continuation  
**Status**: Active (needs task definition - see P0-2)  
**Completed Stages**:
- ✅ Stage 0: Legacy Asset Assessment
- ✅ Stage 1: Specification Integration
- ✅ Stage 2: Data Migration
- ✅ Stage 3: Reproducibility Verification

**Next Steps**: Awaiting Project Owner input on Stage 4 research priorities (baseline reproduction, new experiments, or paper drafting).

---

## 📞 Support

- **DGSF Technical Issues**: Check `repo/docs/` or GitHub Issues
- **AI Workflow OS Integration**: See `docs/plans/EXECUTION_PLAN_DGSF_V1.md`
- **Governance Questions**: Refer to `specs/PROJECT_DGSF.yaml`

---

**Last Updated**: 2026-02-02  
**Maintained by**: AI Workflow OS + DGSF Research Team
