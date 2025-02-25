# DeepSeek RAG System

一个基于 DeepSeek 的高级检索增强生成(RAG)系统，提供模块化、高性能的知识库问答能力。

## 特性

- **模块化设计**

  - 文档处理：支持 PDF、文本、图片等多格式文档
  - 向量存储：基于 FAISS 的高性能向量索引
  - 混合检索：结合语义和关键词的智能检索
  - 流式响应：支持大规模文本生成

- **高性能**

  - GPU 加速支持
  - 批处理优化
  - 异步处理
  - 资源自动管理

- **可扩展**
  - 配置驱动
  - 组件可替换
  - 状态持久化
  - 完整指标收集

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 基础用法

```python
import asyncio
from pathlib import Path
from src.rag import RAGConfig, RAGSystem
async def main():
# 初始化系统
config = RAGConfig(
embedding_model="BAAI/bge-large-zh",
llm_model="deepseek-ai/deepseek-llm-7b-chat"
)
rag = RAGSystem(config)
# 处理文档
await rag.process_documents([Path("your_doc.pdf")])
# 查询
result = await rag.query("你的问题")
print(result["answer"])
asyncio.run(main())
```

### 高级用法

```python
# 自定义配置
config = RAGConfig(
embedding_model="BAAI/bge-large-zh",
llm_model="deepseek-ai/deepseek-llm-7b-chat",
chunk_size=500,
device="cuda:0"
)

# 流式输出
async for token in rag.query_stream("复杂问题"):
print(token, end="", flush=True)
```

## 项目结构

```bash
src/rag/
├── config.py # 配置管理
├── system.py # 系统主类
├── document/ # 文档处理
├── vector_store/ # 向量存储
├── retrieval/ # 检索模块
├── query/ # 查询处理
└── utils/ # 工具模块
```

## 配置选项

| 参数            | 说明           | 默认值                           |
| --------------- | -------------- | -------------------------------- |
| embedding_model | Embedding 模型 | BAAI/bge-large-zh                |
| llm_model       | LLM 模型       | deepseek-ai/deepseek-llm-7b-chat |
| chunk_size      | 文档分块大小   | 1000                             |
| chunk_overlap   | 分块重叠大小   | 100                              |
| device          | 运行设备       | cuda if available else cpu       |

## 开发指南

### 测试

```bash
运行单元测试
pytest tests/test_rag_system.py

运行集成测试
pytest tests/test_integration.py -m integration
```

### 性能评估

系统内置性能指标收集：

- 查询延迟
- 检索准确率
- 资源使用率
- 处理时间

## 注意事项

- 需要安装 Tesseract-OCR 支持图像处理
- 建议使用 GPU 加速大规模文档处理
- 首次运行会下载模型，请确保网络连接

## License

MIT
