# Issue #1546 Workflow

1) Issue 链接与摘要
- URL: https://github.com/THUDM/slime/issues/1546
- 现象：从 checkpoint 恢复训练时，LR scheduler 的步数被重复推进。
- 期望：恢复后 scheduler 保持 checkpoint 中的真实位置，不应二次累加。

2) 根因分析
- `initialize_model_and_optimizer()` 中 `load_checkpoint(...)` 之后再次执行了：
  - `opt_param_scheduler.step(increment=iteration * args.global_batch_size)`
- Megatron 的 checkpoint 恢复路径已经会恢复 scheduler 状态；这里再次 `step` 会导致双计数。

3) 修改清单
- `slime/backends/megatron_utils/model.py`
  - 删除 `opt_param_scheduler.step(increment=iteration * args.global_batch_size)`。

4) 如何验证
- 本机最小验证（CPU 可行）：确认修复行已移除并做语法检查。
  - `rg -n "opt_param_scheduler\\.step\\(increment=iteration \\* args\\.global_batch_size\\)" slime/backends/megatron_utils/model.py`
  - `python3 -m compileall slime/backends/megatron_utils/model.py`
- GPU 环境建议补充：从同一 checkpoint 启动一次恢复训练，检查恢复前后 scheduler `num_steps` 是否一致且不再翻倍。

5) 可选 PR 草稿
- 标题：Fix LR scheduler double-step on checkpoint resume
- Body：Remove redundant scheduler step after `load_checkpoint` in Megatron backend initialization.
