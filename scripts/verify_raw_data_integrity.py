#!/usr/bin/env python3
"""
Verify Raw Data Integrity

校验 projects/dgsf/data/raw/ 目录下所有文件的 SHA256 哈希值，
确保原始数据未被修改。

Usage:
    # 生成/更新 checksums
    python scripts/verify_raw_data_integrity.py --generate
    
    # 验证完整性
    python scripts/verify_raw_data_integrity.py
    
    # 指定数据目录
    python scripts/verify_raw_data_integrity.py --data-dir /path/to/data/raw
    
Exit codes:
    0: 验证通过（或生成成功）
    1: 验证失败
    2: Checksums 文件不存在
"""

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / 'projects' / 'dgsf' / 'data' / 'raw'
CHECKSUMS_FILENAME = '.checksums.sha256'


def sha256_file(filepath: Path) -> str:
    """计算文件的 SHA256 哈希值。"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_data_files(data_dir: Path) -> list[Path]:
    """获取目录下所有数据文件（排除隐藏文件和 checksums 文件）。"""
    files = []
    for f in data_dir.rglob('*'):
        if f.is_file() and not f.name.startswith('.'):
            files.append(f)
    return sorted(files)


def generate_checksums(data_dir: Path) -> dict[str, str]:
    """生成所有文件的 checksums。"""
    checksums = {}
    files = get_data_files(data_dir)
    
    for f in files:
        rel_path = f.relative_to(data_dir).as_posix()
        file_hash = sha256_file(f)
        checksums[rel_path] = file_hash
        print(f"  {rel_path}: {file_hash[:16]}...")
    
    return checksums


def save_checksums(data_dir: Path, checksums: dict[str, str]) -> None:
    """保存 checksums 到文件。"""
    checksums_path = data_dir / CHECKSUMS_FILENAME
    with open(checksums_path, 'w', encoding='utf-8') as f:
        for rel_path, file_hash in sorted(checksums.items()):
            f.write(f"{file_hash}  {rel_path}\n")
    print(f"\nChecksums saved to: {checksums_path}")


def load_checksums(data_dir: Path) -> dict[str, str]:
    """从文件加载 checksums。"""
    checksums_path = data_dir / CHECKSUMS_FILENAME
    if not checksums_path.exists():
        return None
    
    checksums = {}
    with open(checksums_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('  ', 1)
                if len(parts) == 2:
                    file_hash, rel_path = parts
                    checksums[rel_path] = file_hash
    return checksums


def verify_checksums(data_dir: Path) -> tuple[bool, list[str]]:
    """
    验证所有文件的 checksums。
    
    Returns:
        (success: bool, errors: list of error messages)
    """
    stored = load_checksums(data_dir)
    if stored is None:
        return False, [f"Checksums file not found: {data_dir / CHECKSUMS_FILENAME}"]
    
    errors = []
    current_files = {f.relative_to(data_dir).as_posix() for f in get_data_files(data_dir)}
    stored_files = set(stored.keys())
    
    # 检查缺失文件
    for missing in stored_files - current_files:
        errors.append(f"MISSING: {missing}")
    
    # 检查新增文件
    for added in current_files - stored_files:
        errors.append(f"UNEXPECTED: {added} (not in checksums)")
    
    # 验证已有文件
    for rel_path in current_files & stored_files:
        filepath = data_dir / rel_path
        expected = stored[rel_path]
        actual = sha256_file(filepath)
        
        if expected != actual:
            errors.append(f"MODIFIED: {rel_path}")
            errors.append(f"  Expected: {expected}")
            errors.append(f"  Actual:   {actual}")
        else:
            print(f"  ✅ {rel_path}")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Verify raw data integrity")
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate checksums file (overwrites existing)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help='Path to raw data directory'
    )
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Data directory does not exist: {args.data_dir}", file=sys.stderr)
        print("This is expected if no raw data has been added yet.", file=sys.stderr)
        sys.exit(0)
    
    print("=" * 60)
    print(f"RAW DATA INTEGRITY {'GENERATION' if args.generate else 'VERIFICATION'}")
    print(f"Directory: {args.data_dir}")
    print("=" * 60)
    print()
    
    if args.generate:
        checksums = generate_checksums(args.data_dir)
        if checksums:
            save_checksums(args.data_dir, checksums)
            print(f"\n✅ Generated checksums for {len(checksums)} files")
        else:
            print("\n⚠️  No data files found in directory")
        sys.exit(0)
    else:
        success, errors = verify_checksums(args.data_dir)
        
        if success:
            print(f"\n✅ All files verified successfully")
            sys.exit(0)
        else:
            print(f"\n❌ VERIFICATION FAILED")
            print()
            for err in errors:
                print(f"  {err}")
            print()
            print("Run with --generate to create new checksums (if changes are intentional)")
            sys.exit(1)


if __name__ == '__main__':
    main()
