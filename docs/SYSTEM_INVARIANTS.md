# System Invariants（系统不变量）

**文档ID**: SYSTEM_INVARIANTS  
**创建日期**: 2026-02-02  
**目的**: 形式化系统关键不变量，提供可验证的稳定性与治理保障。

---

## INV-1: Task Status State Machine（任务状态机）
**定义**: 任务状态转换必须严格遵循 state_machine.yaml 中的 transitions。  
**验证方法**: 运行 scripts/verify_state_transitions.py（待实现）。  
**违规后果**: 任务生命周期不一致，治理规则失效。  
**参考**: [kernel/state_machine.yaml](../kernel/state_machine.yaml)

## INV-2: WIP Limit（在制品数量上限）
**定义**: 同时处于 running 状态的任务数 ≤ 3。  
**验证方法**: 读取 [configs/gates.yaml](../configs/gates.yaml) 中 wip_limits.max_running_tasks，并对 state/tasks.yaml 进行统计。  
**违规后果**: 上下文切换成本上升，交付周期延长。  
**参考**: [configs/gates.yaml](../configs/gates.yaml), [state/tasks.yaml](../state/tasks.yaml)

## INV-3: YAML Atomicity（YAML写入原子性）
**定义**: 所有 state/*.yaml 的写入必须具备原子性（写临时文件后原子替换）。  
**验证方法**: 代码审查 + 并发测试（test_state_store_concurrency.py）。  
**违规后果**: 数据损坏或部分写入导致状态不一致。  
**参考**: [kernel/state_store.py](../kernel/state_store.py)

## INV-4: State Timestamp Monotonicity（事件时间单调）
**定义**: 同一任务的事件时间必须非递减（后续事件时间 ≥ 前序事件时间）。  
**验证方法**: 在 state/tasks.yaml 的 events 序列中检查时间戳单调性。  
**违规后果**: 审计与回放失真，无法准确定位任务演进。  
**参考**: [state/tasks.yaml](../state/tasks.yaml)

## INV-5: Audit Completeness（审计完整性）
**定义**: 每次任务状态变更必须写入审计日志。  
**验证方法**: 对比 state/tasks.yaml 事件数与审计记录数（脚本待实现）。  
**违规后果**: 审计链断裂，无法追踪责任与变化。  
**参考**: [kernel/audit.py](../kernel/audit.py), [kernel/state_store.py](../kernel/state_store.py)

## INV-6: Path Canonicalization（路径统一）
**定义**: 所有路径必须通过 kernel/paths.py 获取，不允许硬编码绝对路径。  
**验证方法**: 静态搜索硬编码路径 + 单元测试覆盖。  
**违规后果**: 跨环境不一致，部署失败。  
**参考**: [kernel/paths.py](../kernel/paths.py)

## INV-7: Task Priority Validity（任务优先级合法）
**定义**: 任务优先级必须属于 PRIORITY_LEVELS 定义的集合。  
**验证方法**: task_parser.validate_taskcard() 与 state/tasks.yaml 扫描。  
**违规后果**: 排期与治理流程混乱。  
**参考**: [kernel/task_parser.py](../kernel/task_parser.py)

## INV-8: Gate Configuration Integrity（Gate配置完整性）
**定义**: configs/gates.yaml 必须包含 G1-G6 定义与对应检查项。  
**验证方法**: 配置解析 + Gate脚本执行（run_gate_g*.py）。  
**违规后果**: 治理门禁缺失导致低质量产出。  
**参考**: [configs/gates.yaml](../configs/gates.yaml)

## INV-9: MCP Interface Stability（MCP接口一致性）
**定义**: MCP Server公开的工具接口必须与 mcp_server_manifest.json 一致。  
**验证方法**: 运行接口对比脚本（待实现）。  
**违规后果**: Agent调用失败或权限绕过。  
**参考**: [kernel/mcp_server.py](../kernel/mcp_server.py), [mcp_server_manifest.json](../mcp_server_manifest.json)

## INV-10: Stdio Transport Separation（输出通道分离）
**定义**: MCP stdio协议要求 stdout 仅用于协议消息，日志必须写入 stderr。  
**验证方法**: 运行 mcp_stdio 并检查 stdout/stderr 分离。  
**违规后果**: 协议解析失败，Agent通信中断。  
**参考**: [kernel/mcp_stdio.py](../kernel/mcp_stdio.py)

---

## 验证清单（简化）
- [ ] 至少10个不变量已定义
- [ ] 每个不变量包含：定义 / 验证方法 / 违规后果
- [ ] 所有参考链接可访问
