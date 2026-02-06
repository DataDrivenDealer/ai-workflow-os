"""
Config Path Migration Script
============================
Migrates all config path references in the DGSF submodule after the
configs/ flat → subdirectory reorganization.

Categories:
  configs/data_eng/   - de{N}_*.yaml files
  configs/loaders/    - *_loader*.yaml files
  configs/models/     - mask_engine*, sdf_*, rolling_*, ea_*.yaml files
  configs/dev/        - dev_*, de7_dev_test.yaml files
  configs/experiments/ - paneltree_*.yaml files
  configs/schemas/    - *.schema.json, *_required_fields.json files
  configs/pipeline/   - config.yaml (main app config)

Usage:
  python scripts/migrate_config_paths.py --dry-run   # Preview changes
  python scripts/migrate_config_paths.py              # Apply changes
"""

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent / "projects" / "dgsf" / "repo"

# ── Replacement rules ──────────────────────────────────────────────
# Order matters — more specific patterns first to avoid partial matches.
# Each rule: (old_pattern, new_pattern, description)
# These are literal string replacements applied line-by-line.

REPLACEMENTS = [
    # ── DEV (must come before data_eng to avoid configs/dev_small matching configs/de) ──
    ("configs/de7_dev_test.yaml", "configs/dev/de7_dev_test.yaml", "dev config"),
    ("configs/dev_small_universe_stylecube.yaml", "configs/dev/dev_small_universe_stylecube.yaml", "dev config"),
    ("configs/dev_small_universe.yaml", "configs/dev/dev_small_universe.yaml", "dev config"),
    ("configs/dev_small_ea_research.yaml", "configs/dev/dev_small_ea_research.yaml", "dev config"),
    ("configs/dev_small_research.yaml", "configs/dev/dev_small_research.yaml", "dev config"),
    ("configs/dev_small_rolling.yaml", "configs/dev/dev_small_rolling.yaml", "dev config"),

    # ── LOADERS (before data_eng, since some loader names start with fina_) ──
    ("configs/raw_loader_full.yaml", "configs/loaders/raw_loader_full.yaml", "loader config"),
    ("configs/raw_loader.yaml", "configs/loaders/raw_loader.yaml", "loader config"),
    ("configs/macro_loader_full.yaml", "configs/loaders/macro_loader_full.yaml", "loader config"),
    ("configs/fina_indicator_loader_full.yaml", "configs/loaders/fina_indicator_loader_full.yaml", "loader config"),
    ("configs/fina_indicator_loader.yaml", "configs/loaders/fina_indicator_loader.yaml", "loader config"),
    ("configs/fina_loader_full.yaml", "configs/loaders/fina_loader_full.yaml", "loader config"),
    ("configs/fina_loader.yaml", "configs/loaders/fina_loader.yaml", "loader config"),
    ("configs/micro_loader_full.yaml", "configs/loaders/micro_loader_full.yaml", "loader config"),
    ("configs/micro_loader.yaml", "configs/loaders/micro_loader.yaml", "loader config"),
    ("configs/factor_panel_loader_full.yaml", "configs/loaders/factor_panel_loader_full.yaml", "loader config"),
    ("configs/factor_panel_loader.yaml", "configs/loaders/factor_panel_loader.yaml", "loader config"),
    ("configs/xfactor_clean_loader_full.yaml", "configs/loaders/xfactor_clean_loader_full.yaml", "loader config"),
    ("configs/xfactor_clean_loader.yaml", "configs/loaders/xfactor_clean_loader.yaml", "loader config"),

    # ── MODELS ──
    ("configs/mask_engine_full.yaml", "configs/models/mask_engine_full.yaml", "model config"),
    ("configs/mask_engine_dev.yaml", "configs/models/mask_engine_dev.yaml", "model config"),
    ("configs/mask_engine.yaml", "configs/models/mask_engine.yaml", "model config"),
    ("configs/sdf_dev_small.yaml", "configs/models/sdf_dev_small.yaml", "model config"),
    ("configs/rolling_smoke_paneltree_only.yaml", "configs/models/rolling_smoke_paneltree_only.yaml", "model config"),
    ("configs/rolling_10y_paneltree_v2.yaml", "configs/models/rolling_10y_paneltree_v2.yaml", "model config"),
    ("configs/ea_dev_small.yaml", "configs/models/ea_dev_small.yaml", "model config"),

    # ── EXPERIMENTS ──
    ("configs/paneltree_rolling.yaml", "configs/experiments/paneltree_rolling.yaml", "experiment config"),
    ("configs/paneltree_dev_small_universe.yaml", "configs/experiments/paneltree_dev_small_universe.yaml", "experiment config"),
    ("configs/paneltree_full_universe.yaml", "configs/experiments/paneltree_full_universe.yaml", "experiment config"),

    # ── PIPELINE (main app config — very specific match) ──
    ("configs/config.yaml", "configs/pipeline/config.yaml", "pipeline config"),

    # ── DATA_ENG (de{N}_*.yaml — after dev_ patterns are handled) ──
    ("configs/de10_fullwindow.yaml", "configs/data_eng/de10_fullwindow.yaml", "data_eng config"),
    ("configs/de1_a0_candidate_universe.yaml", "configs/data_eng/de1_a0_candidate_universe.yaml", "data_eng config"),
    ("configs/de3_a0.yaml", "configs/data_eng/de3_a0.yaml", "data_eng config"),
    ("configs/de4_a0.yaml", "configs/data_eng/de4_a0.yaml", "data_eng config"),
    ("configs/de5_a0.yaml", "configs/data_eng/de5_a0.yaml", "data_eng config"),
    ("configs/de6_market_structure.yaml", "configs/data_eng/de6_market_structure.yaml", "data_eng config"),
    ("configs/de6_a0.yaml", "configs/data_eng/de6_a0.yaml", "data_eng config"),
    ("configs/de7_fullwindow.yaml", "configs/data_eng/de7_fullwindow.yaml", "data_eng config"),
    ("configs/de7_style_spreads.yaml", "configs/data_eng/de7_style_spreads.yaml", "data_eng config"),
    ("configs/de8_fullwindow.yaml", "configs/data_eng/de8_fullwindow.yaml", "data_eng config"),
    ("configs/de9_a0_stub.yaml", "configs/data_eng/de9_a0_stub.yaml", "data_eng config"),
]

# ── Special cases: regex-based replacements ──────────────────────
# For the schema dynamic path construction (DEFAULT_CONFIGS_DIR)
REGEX_REPLACEMENTS = [
    # de3_fina_eff_builder.py: DEFAULT_CONFIGS_DIR = "configs" → "configs/schemas"
    {
        "file_pattern": "de3_fina_eff_builder.py",
        "old": r'DEFAULT_CONFIGS_DIR\s*=\s*"configs"',
        "new": 'DEFAULT_CONFIGS_DIR = "configs/schemas"',
        "desc": "schema dir (eff_builder)",
    },
    # de3_fina_downloader.py: configs_dir: str = "configs" → "configs/schemas"
    {
        "file_pattern": "de3_fina_downloader.py",
        "old": r'configs_dir:\s*str\s*=\s*"configs"',
        "new": 'configs_dir: str = "configs/schemas"',
        "desc": "schema dir (downloader)",
    },
]

# File extensions to process
EXTENSIONS = {".py", ".yaml", ".yml", ".md", ".ps1", ".sh", ".json", ".toml", ".cfg"}


def find_files(root: Path) -> list[Path]:
    """Find all text files in the repo that might contain config paths."""
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in EXTENSIONS:
            # Skip __pycache__, .git, node_modules, data dirs
            parts = p.parts
            if any(skip in parts for skip in ("__pycache__", ".git", "node_modules", "data", ".mypy_cache")):
                continue
            files.append(p)
    return sorted(files)


def migrate_file(filepath: Path, dry_run: bool = True) -> list[dict]:
    """Apply all replacements to a single file. Returns list of changes."""
    changes = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return changes

    new_content = content
    rel_path = filepath.relative_to(REPO_ROOT)

    # Apply literal replacements
    for old, new, desc in REPLACEMENTS:
        if old in new_content:
            count = new_content.count(old)
            new_content = new_content.replace(old, new)
            changes.append({
                "file": str(rel_path),
                "old": old,
                "new": new,
                "count": count,
                "desc": desc,
            })

    # Apply regex replacements (file-specific)
    for rule in REGEX_REPLACEMENTS:
        if rule["file_pattern"] in filepath.name:
            matches = re.findall(rule["old"], new_content)
            if matches:
                new_content = re.sub(rule["old"], rule["new"], new_content)
                changes.append({
                    "file": str(rel_path),
                    "old": rule["old"],
                    "new": rule["new"],
                    "count": len(matches),
                    "desc": rule["desc"],
                })

    if changes and not dry_run:
        filepath.write_text(new_content, encoding="utf-8")

    return changes


def main():
    parser = argparse.ArgumentParser(description="Migrate DGSF config paths after reorganization")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    if not REPO_ROOT.exists():
        print(f"ERROR: Repo root not found: {REPO_ROOT}")
        return 1

    print(f"{'DRY RUN' if args.dry_run else 'APPLYING'} config path migration")
    print(f"Repo root: {REPO_ROOT}")
    print()

    files = find_files(REPO_ROOT)
    print(f"Scanning {len(files)} files...\n")

    all_changes = []
    files_changed = set()

    for f in files:
        changes = migrate_file(f, dry_run=args.dry_run)
        if changes:
            all_changes.extend(changes)
            for c in changes:
                files_changed.add(c["file"])

    # Print summary grouped by category
    if all_changes:
        print("=" * 80)
        print(f"{'WOULD CHANGE' if args.dry_run else 'CHANGED'} {len(files_changed)} files, {sum(c['count'] for c in all_changes)} replacements")
        print("=" * 80)

        # Group by file
        by_file = {}
        for c in all_changes:
            by_file.setdefault(c["file"], []).append(c)

        for filepath, changes in sorted(by_file.items()):
            print(f"\n  {filepath}:")
            for c in changes:
                print(f"    [{c['desc']}] x{c['count']}: {c['old']} → {c['new']}")

        print(f"\n{'=' * 80}")
        # Summary by category
        by_desc = {}
        for c in all_changes:
            by_desc.setdefault(c["desc"], 0)
            by_desc[c["desc"]] += c["count"]
        print("By category:")
        for desc, count in sorted(by_desc.items()):
            print(f"  {desc}: {count} replacements")
    else:
        print("No changes needed — all paths already up to date.")

    return 0


if __name__ == "__main__":
    exit(main())
