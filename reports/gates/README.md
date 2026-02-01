# Gate Reports Directory

This directory stores gate verification reports generated during CI/CD and manual checks.

## Report Types

- `G1_*.md` - Data Quality gate reports
- `G2_*.md` - Sanity Check gate reports  
- `G3_*.md` - Performance & Robustness gate reports
- `G4_*.md` - Approval gate reports
- `G5_*.md` - Live Safety gate reports
- `ci_*.md` - Full CI pipeline reports

## Generating Reports

```bash
# Generate specific gate report
python scripts/gate_check.py --gate G2 --output markdown > reports/gates/G2_$(date +%Y%m%d).md

# Generate full CI report
python scripts/ci_gate_reporter.py --format markdown > reports/gates/ci_$(date +%Y%m%d).md

# Validate TaskCard gate completion
python scripts/taskcard_gate_validator.py tasks/TASK_XXX.md
```

## Report Retention

- CI reports: 30 days
- Gate pass evidence: Permanent (linked from TaskCards)
