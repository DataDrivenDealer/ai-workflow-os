#!/usr/bin/env python3
"""
DGSF Full Data Engineering Pipeline Runner
==========================================
自动顺序执行 DE3 → DE5 → DE7，完成/失败时弹窗+声音提醒

用法:
  python scripts/run_full_pipeline.py

预计总运行时间: 6-12 小时 (建议夜间执行)
"""

import os
import sys
import time
import subprocess
import ctypes
from pathlib import Path
from datetime import datetime

# === 配置 ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_FULL = PROJECT_ROOT / "data" / "full"
LOG_FILE = PROJECT_ROOT / "data" / "pipeline_run.log"

# Windows 提醒函数
def notify_windows(title: str, message: str, is_error: bool = False):
    """Windows 弹窗 + 声音提醒"""
    import winsound
    
    # 播放系统声音
    if is_error:
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
    else:
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    
    # 弹窗提醒
    icon = 0x10 if is_error else 0x40  # MB_ICONERROR or MB_ICONINFORMATION
    ctypes.windll.user32.MessageBoxW(0, message, title, icon | 0x1000)  # MB_SYSTEMMODAL

def log(msg: str):
    """记录日志到文件和控制台"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def check_env():
    """检查环境变量"""
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        log("ERROR: TUSHARE_TOKEN 未设置！")
        notify_windows("DGSF Pipeline 错误", "TUSHARE_TOKEN 环境变量未设置！\n请设置后重新运行。", is_error=True)
        sys.exit(1)
    log(f"TUSHARE_TOKEN 已设置 (长度: {len(token)})")

def run_stage(name: str, script: str, expected_output: Path, expected_min_rows: int) -> bool:
    """运行单个阶段"""
    log(f"\n{'='*60}")
    log(f"开始执行: {name}")
    log(f"脚本: {script}")
    log(f"预期输出: {expected_output}")
    log(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 执行脚本
        result = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_ROOT,
            capture_output=False,  # 直接输出到控制台
            text=True,
            timeout=8 * 3600  # 8小时超时
        )
        
        elapsed = time.time() - start_time
        log(f"{name} 脚本执行完成，耗时: {elapsed/60:.1f} 分钟")
        
        if result.returncode != 0:
            log(f"ERROR: {name} 返回码 {result.returncode}")
            return False
        
        # 验证输出文件
        if expected_output.exists():
            import pandas as pd
            df = pd.read_parquet(expected_output)
            rows = len(df)
            log(f"{name} 输出验证: {rows:,} 行")
            
            if rows < expected_min_rows:
                log(f"WARNING: 行数 {rows:,} 低于预期 {expected_min_rows:,}")
                return False
            
            return True
        else:
            log(f"ERROR: 输出文件不存在: {expected_output}")
            return False
            
    except subprocess.TimeoutExpired:
        log(f"ERROR: {name} 执行超时 (8小时)")
        return False
    except Exception as e:
        log(f"ERROR: {name} 执行异常: {e}")
        return False

def run_de7_factor_panel() -> bool:
    """运行 DE7 因子面板构建"""
    log(f"\n{'='*60}")
    log(f"开始执行: DE7 Factor Panel")
    log(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 添加 repo/src 到 Python 路径
        repo_src = PROJECT_ROOT / "repo" / "src"
        sys.path.insert(0, str(repo_src))
        
        from dgsf.dataeng.de7_factor_panel import FactorPanelBuilder
        
        builder = FactorPanelBuilder(
            data_dir=DATA_RAW,
            output_dir=DATA_FULL
        )
        
        panel = builder.build()
        
        elapsed = time.time() - start_time
        log(f"DE7 完成，耗时: {elapsed/60:.1f} 分钟")
        log(f"Factor panel: {len(panel):,} 行, {len(panel.columns)} 列")
        
        return True
        
    except ImportError as e:
        log(f"DE7 模块导入失败: {e}")
        log("尝试备用方案: 直接运行 de7_factor_panel_a0_runner.py")
        
        # 备用方案
        runner = PROJECT_ROOT / "repo" / "src" / "dgsf" / "dataeng" / "de7_factor_panel_a0_runner.py"
        if runner.exists():
            result = subprocess.run(
                [sys.executable, str(runner)],
                cwd=PROJECT_ROOT,
                capture_output=False,
                text=True
            )
            return result.returncode == 0
        return False
        
    except Exception as e:
        log(f"ERROR: DE7 执行异常: {e}")
        import traceback
        log(traceback.format_exc())
        return False

def main():
    """主流程"""
    log("\n" + "="*70)
    log("DGSF DATA ENGINEERING FULL PIPELINE")
    log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*70)
    
    # 检查环境
    check_env()
    
    # 检查已完成的阶段
    de1_done = (DATA_RAW / "daily_prices.parquet").exists()
    de2_done = (DATA_RAW / "macro_monthly.parquet").exists()
    
    if not de1_done or not de2_done:
        log("ERROR: DE1 或 DE2 未完成，无法继续")
        notify_windows("DGSF Pipeline 错误", "DE1/DE2 数据未就绪！\n请先完成 DE1/DE2。", is_error=True)
        sys.exit(1)
    
    log("✓ DE1 (Daily Prices) 已完成")
    log("✓ DE2 (Macro Monthly) 已完成")
    
    # 定义执行计划
    stages = [
        {
            "name": "DE3 Financial Indicators",
            "script": "scripts/de3_financial_loader.py",
            "output": DATA_RAW / "fina_indicator.parquet",
            "min_rows": 100000,
            "skip_if_exists": True,
        },
        {
            "name": "DE5 Microstructure",
            "script": "scripts/de5_microstructure_loader.py",
            "output": DATA_RAW / "daily_basic.parquet",
            "min_rows": 5000000,
            "skip_if_exists": True,
        },
    ]
    
    failed_stage = None
    
    for stage in stages:
        name = stage["name"]
        output = stage["output"]
        
        # 检查是否已完成
        if stage["skip_if_exists"] and output.exists():
            import pandas as pd
            df = pd.read_parquet(output)
            if len(df) >= stage["min_rows"]:
                log(f"✓ {name} 已完成 ({len(df):,} 行), 跳过")
                continue
        
        # 执行阶段
        success = run_stage(
            name=name,
            script=stage["script"],
            expected_output=output,
            expected_min_rows=stage["min_rows"]
        )
        
        if not success:
            failed_stage = name
            break
    
    # DE7 (需要特殊处理)
    if not failed_stage:
        de7_output = DATA_FULL / "de7_factor_panel.parquet"
        if de7_output.exists():
            log("✓ DE7 Factor Panel 已存在, 跳过")
        else:
            success = run_de7_factor_panel()
            if not success:
                failed_stage = "DE7 Factor Panel"
    
    # 最终报告
    log("\n" + "="*70)
    log("PIPELINE 执行完毕")
    log(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*70)
    
    if failed_stage:
        log(f"❌ 失败阶段: {failed_stage}")
        notify_windows(
            "DGSF Pipeline 失败",
            f"数据工程管道在 {failed_stage} 阶段失败！\n\n请检查日志: {LOG_FILE}",
            is_error=True
        )
        sys.exit(1)
    else:
        log("✅ 全部阶段完成!")
        
        # 打印最终数据摘要
        import pandas as pd
        log("\n📊 数据摘要:")
        for name, path in [
            ("daily_prices", DATA_RAW / "daily_prices.parquet"),
            ("adj_factor", DATA_RAW / "adj_factor.parquet"),
            ("macro_monthly", DATA_RAW / "macro_monthly.parquet"),
            ("fina_indicator", DATA_RAW / "fina_indicator.parquet"),
            ("daily_basic", DATA_RAW / "daily_basic.parquet"),
        ]:
            if path.exists():
                df = pd.read_parquet(path)
                log(f"  {name}: {len(df):,} 行")
        
        notify_windows(
            "DGSF Pipeline 完成 ✅",
            f"数据工程管道全部完成！\n\n日志: {LOG_FILE}\n\n可以开始 DE7 因子面板构建。",
            is_error=False
        )


if __name__ == "__main__":
    main()
