---
title: NanoBot 项目研究索引
description: NanoBot 开源 AI Agent 框架研究笔记索引
tags: [index, ai-agent, nanobot]
created: 2026-05-02
---

# NanoBot 项目研究索引

> 来源: [HKUDS/nanobot](https://github.com/HKUDS/nanobot) | 版本: v0.1.5.post3 | 许可: MIT

## 笔记导航

- [[nanobot-project-analysis|项目深度分析]] — 特征、难点、学习路径、代码结构

## 速览卡片

| 维度 | 摘要 |
|------|------|
| **定位** | 超轻量级开源 AI Agent，研究友好 |
| **语言** | Python 3.11+ (后端) + TypeScript (WebUI) |
| **核心文件** | `agent/runner.py` (~1100 行) |
| **提供商** | 12+ LLM providers |
| **聊天频道** | 15+ channels |
| **依赖** | `openai`, `anthropic`, `pydantic`, `httpx`, `websockets` 等 |
| **测试** | 100+ 测试文件，pytest + pytest-asyncio |

## 核心模块地图

```
agent/loop.py          ← 消息分发 & 并发控制
agent/runner.py        ← ★ 核心 Agent 执行循环
agent/context.py       ← 上下文组装 (系统提示 + 记忆 + 技能)
agent/memory.py        ← 记忆存储 + 压缩 + Dream
providers/base.py      ← LLM 抽象 + 重试策略
channels/base.py       ← 频道接口
bus/queue.py           ← 消息总线 (asyncio.Queue)
```

## 快速学习路径

1. `nanobot.py` → `bus/queue.py` → `agent/runner.py` (核心循环)
2. `agent/loop.py` → `agent/context.py` (消息流 & 上下文)
3. `providers/base.py` → `channels/base.py` (扩展点)
4. `agent/memory.py` → `agent/subagent.py` (高级特性)
