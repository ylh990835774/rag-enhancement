# DeepSeek Advanced RAG System

一个基于 DeepSeek LLM 的高级检索增强生成(RAG)系统,支持多模态文档处理、自适应检索和上下文压缩等特性。

## 主要特性

- 多模态文档处理

  - 支持 PDF、文本、图片等多种格式
  - OCR 文字识别
  - 图像内容理解

- 智能检索

  - 层次化检索架构
  - 查询扩展
  - 文档重排序
  - 自适应检索策略

- 上下文优化

  - 智能上下文压缩
  - 相关性排序
  - 文档去重

- 系统扩展性
  - 模块化设计
  - 可配置组件
  - 状态保存/加载

## 快速开始

### 安装依赖

```bash
pip install langchain torch transformers sentence-transformers pdf2image pytesseract faiss-cpu pillow
```

### 基本使用

```python
from deepseek_rag import DeepSeekAdvancedRAG
```

### 初始化系统

```python
rag = DeepSeekAdvancedRAG()
```

### 添加文档

```python
rag.add_document("document.pdf")
rag.add_document("image.jpg")
```

### 构建知识库

```python
rag.build_knowledge_base()
```

### 查询

```python
result = rag.query("你的问题")
print(result)
```

### 高级功能

```python
# 启用重排序
rag.enable_reranker()

# 带图像的查询
result = rag.query("图表分析", image_path="chart.png")

# 自适应查询
result = rag.adaptive_query("复杂问题")

# 保存/加载系统状态
rag.save("./rag_system")
rag = DeepSeekAdvancedRAG.load("./rag_system")
```

## 系统架构

- DeepSeekKnowledgeBase: 基础知识库管理
- DeepSeekAdvancedRAG: 高级 RAG 系统封装
- QueryExpander: 查询扩展
- ContextCompressor: 上下文压缩
- AdaptiveRAG: 自适应检索策略
- HierarchicalRetrieval: 层次化检索
- DocumentReranker: 文档重排序
- MultimodalRAG: 多模态处理

## 配置选项

- model_name: Embedding 模型路径
- llm_path: LLM 模型路径
- device: 运行设备(CPU/CUDA)
- chunk_size: 文档分块大小
- chunk_overlap: 分块重叠大小

## 性能评估

系统提供内置评估功能:

```python
metrics = rag.evaluate_retrieval(test_queries, ground_truth)
print(metrics)
```

## 注意事项

- 需要安装 Tesseract-OCR 用于图像文字识别
- 大规模文档建议使用 GPU 加速
- 保存路径需要足够存储空间

## License

MIT
