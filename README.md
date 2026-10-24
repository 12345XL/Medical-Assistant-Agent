# Medical Assistant Agent

医疗助手 Agent 项目，包含多智能体医疗助手、MediX-R1 模型训练与评估代码，以及 Agent 记忆系统设计文档。

## 项目目录

| 目录 | 内容 |
| --- | --- |
| [medix-agent-swarm](medix-agent-swarm/) | 医疗多智能体系统：Agent Loop、Agent Swarm、Skills、记忆管理与 Milvus 知识库 |
| [MediX-R1/training](MediX-R1/training/) | 模型训练代码、配置与脚本 |
| [MediX-R1/eval](MediX-R1/eval/) | 模型评估代码 |
| [MediX-R1/eval2](MediX-R1/eval2/) | 另一组评估脚本与配置 |
| [docs](docs/) | Agent 记忆系统设计指南与模板 |

## 配置与运行

先复制配置模板并填写自己的 API 设置：

```bash
cp config.example.py config.py
```

医疗助手的依赖安装、知识库初始化和运行方式见 [详细说明](medix-agent-swarm/README.md)。代码中部分脚本含原开发环境的绝对路径，运行前需要按本机目录调整。训练和评估也需要配置相应的模型、数据路径及运行环境。

## 仓库内容说明

本仓库保存源代码、配置模板、知识库文本和设计文档。本地 API 配置、工具个人设置、数据库、缓存、运行日志、生成结果和演示视频不纳入版本控制。

原有子项目说明中的测试结果为其历史记录，本次上传未重新运行依赖外部 API 或 GPU 的测试。
