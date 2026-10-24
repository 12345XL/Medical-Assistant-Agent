# MediX Agent Swarm

Medical Assistant Agent 的多智能体应用模块，提供命令行交互和 Python 异步调用接口。系统包含健康咨询、症状信息分析、医学资料研究三类专业 Agent，共享 9 个 Skills，并通过任务路由、并行协作和分层记忆完成问答。

整体架构、技术栈及模型训练评估介绍见 [项目首页](../README.md)。

## 运行环境

- Python 3.12；依赖见 [requirements.txt](requirements.txt)。
- 支持 function calling 的 OpenAI-compatible API。
- Milvus Lite 本地知识库，使用 `BAAI/bge-small-zh-v1.5` 生成 Embedding。
- 可选 Mem0 API，用于跨会话摘要保存与检索。
- 可选 Redis，用于短期记忆持久化；默认使用进程内存。

## 安装与启动

以下命令从仓库根目录 `Medical-Assistant-Agent/` 开始执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r medix-agent-swarm/requirements.txt
cp config.example.py config.py
```

在 `config.py` 的 `LLM_CONFIG` 中填写 `api_key`、`model_name` 和 `base_url`。需要长期记忆时填写 `MEM0_CONFIG["api_key"]`，不启用时将其设为空字符串。

```bash
export PYTHONPATH="$PWD:$PWD/medix-agent-swarm${PYTHONPATH:+:$PYTHONPATH}"
cd medix-agent-swarm
python knowledge/scripts/import_hardcoded_data.py
python main.py
```

知识导入脚本读取 `knowledge/data/documents/` 下的文本，写入 Milvus Lite。首次运行需下载 Embedding 模型。后续启动保持工作目录为 `medix-agent-swarm/`，以正确读取相对路径配置和知识库。

| 命令 | 作用 |
| --- | --- |
| `python main.py` | 启动交互式问答 |
| `python main.py --verbose` | 显示详细执行日志 |
| `python examples/test_all.py` | 执行集成测试并生成报告 |

交互界面中，`help` 显示帮助，`clear` 清屏，`exit` / `quit` 退出。一次交互进程中的连续问题使用同一个会话 ID。

## 请求处理流程

1. `SwarmCoordinator` 根据 `session_id` 读取近期对话，并查询相关历史摘要。
2. `LeadAgent` 分析问题，生成子任务及对应的 Agent 分配。
3. 单任务进入相应专业 Agent；多任务进入 Swarm，并行执行分配后的子任务。
4. 每个 Agent 的 `AgentLoop` 将可用 Skills 交给模型，通过 function calling 执行检索或分析。
5. 工具结果回传模型，输出检查与自动修复逻辑处理可修复项。
6. 多任务结果由 LeadAgent 汇总，系统记录对话、执行信息及会话摘要。

Swarm 等待上限为 90 秒，超时时尝试汇总已完成的结果。Agent Loop 支持迭代次数与工具调用次数限制；工具推荐范围检查以告警为主，具体行为见 [约束验证器](constraints/validator.py)。

## Python 调用

```python
import asyncio
from swarm import process_with_swarm

async def main():
    session_id = "example-session"
    first = await process_with_swarm(
        "请整理高血压相关的生活方式管理资料。",
        session_id=session_id,
    )
    print(first["answer"])

    follow_up = await process_with_swarm(
        "请补充这些资料的参考依据。",
        session_id=session_id,
    )
    print(follow_up["answer"])

asyncio.run(main())
```

常用返回字段：

| 字段 | 含义 |
| --- | --- |
| `answer` | 回答正文 |
| `session_id` | 会话标识 |
| `swarm_enabled` | 本次是否进入多 Agent 协作 |
| `suggestions` | 建议条目 |
| `disclaimer` | 输出中的使用范围说明 |
| `agents_involved` | Swarm 模式下参与结果汇总的 Agent |
| `timeout_occurred` | Swarm 模式下是否发生等待超时 |

## 扩展 Skills

技能目录位于 [`.claude/skills/`](.claude/skills/)。每个技能包含 `SKILL.md` 元数据和 `script/` 下的 Python 实现。

1. 参考现有技能添加目录、元数据和实现函数。
2. 按现有约定使用技能目录名对应的下划线函数名，并为参数提供清晰的签名与默认值。
3. `Skill Loader` 自动发现函数，`SkillRegistryMixin` 推断参数并注册到各专业 Agent。
4. 检查 [Agent 约束配置](constraints/agent_constraints.yaml)，补充相应调用建议和输出要求。

当前技能覆盖知识查询、生活方式信息、风险提示、症状分析、疾病编码、指南检索、深度研究、会话历史及相似案例检索。

## 记忆与知识库配置

| 组件 | 配置入口 | 默认行为 |
| --- | --- | --- |
| 模型调用 | 根目录 `config.py` 的 `LLM_CONFIG` | 使用配置的 OpenAI-compatible API |
| 短期记忆 | `ShortTermMemory(storage_type="memory")` | 按会话保存消息；协调器检索最近 10 条 |
| Redis 后端 | `ShortTermMemory(storage_type="redis", redis_config=...)` | 需在协调器初始化短期记忆时选择，默认不启用 |
| Mem0 长期记忆 | 根目录 `config.py` 的 `MEM0_CONFIG` | 有效配置下保存和检索历史摘要 |
| 文本知识 | `knowledge/data/documents/` | 按类型导入并检索 |
| 向量存储 | `MedicalKnowledgeBase` 构造参数 | `knowledge/data/milvus_lite.db` |

短期记忆通过会话 ID 区分对话；当前 Mem0 调用使用固定的 `medix_user` 标识。接入多用户服务时，需要扩展用户标识的传递和存取隔离。

## 测试与执行观察

[examples/test_all.py](examples/test_all.py) 包含 26 个用例，覆盖任务路由、Agent Loop、SharedContext、记忆集成、知识查询、深度研究及约束修复。测试会调用配置的外部模型和服务，执行后输出统计并生成 `TEST_REPORT.md`。

使用 `--verbose` 可查看模型调用、Skill 执行及路由信息；协作过程的数据结构可从 [SharedContext](swarm/shared_context.py) 和 [SessionSummary](memory/session_summary.py) 查看。
