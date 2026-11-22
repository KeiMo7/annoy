# 多租户索引内存管理 — 交付说明

## 问题

容量评审的疑问：N 个租户的 Annoy 索引全部常驻内存，到底占多少？扩容只能加内存。
本交付在**不改动** `AnnoyIndex` 任何行为的前提下，新增一个可选的统一入口
`TenantIndexManager`：按租户挂载索引、总内存预算、LRU 卸载、查询自动重挂。

## 改动文件

| 文件 | 说明 |
|---|---|
| `annoy/tenancy.py` | 新模块：`TenantIndexManager` 及错误类型（纯 stdlib，无新依赖） |
| `annoy/__init__.py` | 追加导出（纯增量；不使用者行为逐条不变） |
| `test/tenancy_test.py` | 21 个新测试 |
| `output/` | 测试输出、样例报告、吞吐对比（本目录） |

## 需求对照

一、形态
1. `register(tenant_id, path, dim, metric)` 统一入口；report 记录每条租户
   n_items / dim / file_size / 估算常驻内存（`resident_bytes`）。
2. `budget_bytes` 可配；超预算按 LRU（last_access 时间戳）卸载；查询时自动重挂。
3. 引用计数（`lease()`/`query()` 内部 pin）；refcount>0 的租户永不被卸载。
4. 并发挂载：`loading` 状态 + 条件变量，文件只加载一次
   （`test_concurrent_mount_loads_exactly_once` 断言 loads==1）。
5. 挂载失败（损坏/维度不符/文件缺失）抛 `TenantMountError`，只影响该租户。
6. `report()`：每租户 hits/misses/hit_rate、remounts、avg_query_ms、resident_bytes。
7. `idle_timeout_s` + 后台 reaper 线程自动释放；释放前记录 `last_access`，
   并记录 `last_unload_at`。
8. 重挂后结果与常驻逐条一致（含距离），并与直接 load 的 AnnoyIndex 一致
   （`test_remount_results_are_identical`）。
9. 不使用 manager 时现有接口行为不变（`test_existing_interface_is_untouched`）；
   本交付未改动 `src/` 任何一行。
10. 每次查询前 stat 文件（mtime_ns+size 签名），替换后下一次查询前必重挂
    （`test_replaced_file_is_remounted_before_next_query`）。
11. `set_budget()` 在线生效，立即按 LRU 收缩，无需重启
    （`test_budget_change_applies_without_restart`）。

二、硬约束
1. 内存估算同源：`estimate_resident_bytes(path)` 是唯一口径，预算记账、淘汰
   决策、report 全部使用它（`test_estimated_memory_matches_accounted_usage`）。
2. 卸载只在 refcount==0 时发生，查询路径无全局锁（每租户一把锁，加载在锁外
   进行），不会读到已释放内存，也不会长尾阻塞。
3. 未改索引文件格式与查询语义（`src/` 零改动）。
4. 无新依赖（仅 os/threading/time/contextlib/collections）。

三、失败与降级
1. 单租户需求 > 总预算：`TenantBudgetError`，携带 required_bytes / available_bytes。
2. 重挂失败抛 `TenantMountError`，绝不返回空结果冒充「无相似项」。
3. manager 从不写盘，卸载是纯内存操作（munmap），退出不留临时文件
   （`test_unload_leaves_no_temp_files`）。

四、既有行为
1. 构建/保存/查询接口语义不变（未触碰）。
2. 多线程吞吐见 `throughput_check.txt`：10 万条索引、4 线程，
   严格模式 +7.9%，interval 模式无下降。
3. 单索引加载后结果与延迟不变（manager 只是把查询转发给原接口）。
4. 既有测试全部通过，见下。

## 测试结果（实际输出见本目录文件）

- `tenancy_test_output.txt`：新增 21 个测试全部通过。
- `existing_tests_output.txt`：全套 124 passed, 1 skipped；
  4 个 failed 均为 `accuracy_test.py` 需联网下载数据集
  （urllib HTTPError），改动前基线即如此，与本次交付无关。
- `sample_report.json`：30 租户、600 次查询、预算在线收缩后的真实 report。
- `throughput_check.txt`：raw vs managed 吞吐对比。
- `tenant_manager_demo.py` / `throughput_check.py`：可重复运行的脚本。

## 构建说明（本环境）

系统无 g++，使用 `zig c++`（pip 包 ziglang）作为编译器；注意
`-march=native` 在 zig 下会产生崩溃的二进制，构建时用了
`ANNOY_COMPILER_ARGS` 去掉 `-march=native`/`-ffast-math`：

```
ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-fpermissive,-O2,-std=c++14,-DANNOYLIB_MULTITHREADED_BUILD" \
  CC="python -m ziglang cc" CXX=<过滤 --exclude-libs 的包装脚本> \
  python setup.py build_ext --inplace
```
