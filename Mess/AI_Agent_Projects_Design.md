# 🏗 AI Agent 应用开发岗 — 实践项目设计方案

> 适用人群：目标 AI Agent 应用开发岗位的求职者  
> 级别：L1（入门）→ L2（基础）→ L3（进阶）→ L4（高阶）  
> 原则：**每个项目独立可展示，难度递进，覆盖不同面试考点**

---

## 项目总览

| 级别 | 项目名称 | 核心考点 | 建议用时 | 简历含金量 |
|------|---------|---------|---------|-----------|
| **L1 🟢** | 智能 RAG 问答助手 | RAG 全流程、文档解析、Prompt 工程 | 4 天 | ⭐⭐⭐ |
| **L2 🔵** | ReAct Agent 工具箱 | Tool Calling、Function Call、Agent 循环 | 4 天 | ⭐⭐⭐ |
| **L3 🟠** | Multi-Agent 代码审查平台 | LangGraph、状态图、多 Agent 编排 | 6 天 | ⭐⭐⭐⭐ |
| **L4 🔴** | Agent 云服务平台 | MCP 协议、生产部署、评测体系、架构设计 | 10 天 | ⭐⭐⭐⭐⭐ |

---

# L1 🟢 智能 RAG 问答助手

> 难度：⭐ 入门级  
> 目标：掌握 RAG 完整流水线，理解检索增强生成的核心概念  
> 面试覆盖：RAG 原理、Chunk 策略、Embedding、检索优化

## 项目简介

用户上传文档（PDF/TXT/Markdown），系统自动解析、分块、向量化存储，用户基于文档内容进行多轮对话。

## 详细设计路线

### 第 1 天 — 文档加载与分块

```
目标：能解析多种格式文档，理解不同分块策略的差异
```

**核心代码模块：`document_loader.py`**

```python
# 要实现的类/函数
class DocumentLoader:
    - load_pdf(file_path)         # PDF 解析
    - load_txt(file_path)         # 纯文本解析  
    - load_markdown(file_path)    # Markdown 解析
    - load_url(url)              # 网页抓取解析

class TextSplitter:
    - recursive_split(docs, chunk_size, chunk_overlap)  # 递归分块
    - semantic_split(docs)        # 语义分块（按段落/标题）
    - token_split(docs)           # 按 Token 数分块
```

**实验：对比不同分块策略**
```
准备一份 10 页的测试文档，对比：
- chunk_size=200 vs 500 vs 1000 对检索效果的影响
- chunk_overlap=0 vs 50 vs 100 对上下文完整性的影响
记录结论写入项目 README
```

**技术栈：** PyMuPDF / pdfplumber, langchain_text_splitters, BeautifulSoup

**产出物：** `document_loader.py` + `splitter.py` + 分块策略对比笔记

---

### 第 2 天 — 向量存储与检索

```
目标：掌握 Embedding + 向量数据库的全流程
```

**核心代码模块：`vector_store.py`**

```python
class VectorStore:
    - create_embeddings(texts)          # 调用 Embedding API
    - store_vectors(docs, persist_dir)  # 存入向量数据库
    - load_vectors(persist_dir)         # 加载已有向量库
    
    - similarity_search(query, k)       # 基础相似度搜索
    - hybrid_search(query, k)           # 混合搜索（向量 + 关键词）
    - mmr_search(query, k)              # MMR 最大边际相关性检索
    - search_with_score(query)          # 带分数返回
```

**支持的 Embedding 方案：**
```
- OpenAI: text-embedding-3-small / text-embedding-3-large
- 本地: BGE-M3 (BAAI/bge-m3)
- 本地: sentence-transformers/all-MiniLM-L6-v2
```

**支持的向量数据库（选一个即可）：**
```
- ChromaDB（本地，最简单）
- FAISS（本地，高性能）
- Milvus / Qdrant（需要 Docker，生产级）
```

**产出物：** `vector_store.py` + `embeddings.py` + 检索效果测试

---

### 第 3 天 — 检索增强生成与对话

```
目标：构建问答链，支持多轮对话
```

**核心代码模块：`qa_chain.py`**

```python
class RAGChain:
    - build_basic_chain(llm, retriever)           # 基础 QA Chain
    - build_conv_chain(llm, retriever, memory)    # 多轮对话 Chain
    - build_with_reranker(llm, retriever, reranker) # 带重排序的 Chain
    
    - ask(question) -> answer + sources            # 单次问答
    - chat(question, history) -> answer            # 多轮对话

class QueryTransformer:
    - rewrite_query(original)      # 查询重写（把对话变成独立查询）
    - hyde_query(original)         # HyDE：先假设答案再检索
    - generate_sub_queries(query)  # 复杂查询拆解为子查询

class Reranker:
    - cohere_rerank(query, docs)   # Cohere 重排序
    - cross_encoder(query, docs)   # Cross-Encoder 重排序
    - keyword_boost(query, docs)   # 关键词加权
```

**产出物：** `qa_chain.py` + `query_transform.py` + `reranker.py`

---

### 第 4 天 — Web UI 与集成

```
目标：构建可交互的 UI，项目整体集成
```

**核心代码模块：`app.py`（Streamlit）**

```python
# UI 功能清单
- 文件上传（支持多文件、拖拽）
- 文档处理进度条
- 对话界面（类似 ChatGPT 的聊天 UI）
- 来源展示（展开可查看检索到的原始段落）
- 历史对话管理
- 系统设置（Chunk 参数、模型选择、TopK 调整）
```

**项目目录结构（最终）：**

```
rag-qa-assistant/
├── app.py                      # Streamlit UI 入口
├── requirements.txt           # 依赖列表
├── .env.example              # 环境变量模板
├── README.md                 # 项目说明
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # 文档加载
│   ├── splitter.py           # 文档分块
│   ├── vector_store.py       # 向量存储
│   ├── embeddings.py         # Embedding 封装
│   ├── qa_chain.py           # 问答链
│   ├── query_transform.py    # 查询优化
│   ├── reranker.py           # 重排序
│   └── config.py             # 配置管理
│
├── data/
│   └── sample_docs/          # 测试文档
│
├── tests/
│   ├── test_loader.py
│   ├── test_splitter.py
│   ├── test_retriever.py
│   └── test_qa.py
│
└── chroma_db/                # 向量数据库持久化目录（运行时生成）
```

**产出物：** 完整可运行的 RAG 问答系统 + GitHub 仓库

---

## L1 面试考点清单

| 问题 | 考察点 | 在项目中的体现 |
|------|--------|--------------|
| Chunk Size 怎么选？ | 分块策略理解 | 项目中做了对比实验 |
| 为什么检索结果不准确？ | 检索质量认知 | Hybrid Search + 重排序 |
| 多轮对话怎么做？ | 对话历史管理 | Conversation Memory |
| 怎么评估 RAG 质量？ | 评测意识 | 可扩展评估脚本 |
| 处理过什么边缘情况？ | 问题解决能力 | 文档为空/检索不到/长文档 |

---

# L2 🔵 ReAct Agent 工具箱

> 难度：⭐⭐ 基础级  
> 目标：深入理解 Agent 核心机制，从零实现 ReAct 循环  
> 面试覆盖：Tool Calling 原理、Function Call 解析、Agent 循环控制

## 项目简介

一个可扩展的 Agent 工具箱，注册一系列工具（计算、搜索、文件操作、API 调用），Agent 根据用户需求自动选择并调用工具完成任务。

## 详细设计路线

### 第 1 天 — 手写 ReAct Agent 核心引擎

```
目标：不依赖框架，从零实现 ReAct 循环
```

**核心代码：`core/react_agent.py`**

```python
class ReActAgent:
    """
    ReAct 循环核心：
    
    1. Thought: 分析用户输入，决定下一步行动
    2. Action:  选择一个工具并生成参数
    3. Observation: 获取工具执行结果
    4. 重复 1-3，直到可以给出 Final Answer
    """
    
    def __init__(self, llm, tools):
        self.llm = llm          # LLM 实例（支持 tool calling）
        self.tools = tools      # 工具注册表
        self.max_steps = 10     # 最大步数
        self.memory = []        # 对话记忆
    
    def register_tool(self, tool):
        """注册工具：工具名、描述、参数 Schema、执行函数"""
    
    def _build_system_prompt(self):
        """构建 System Prompt，定义 ReAct 格式"""
    
    def _parse_tool_call(self, response):
        """解析 LLM 返回的工具调用"""
    
    def _execute_tool(self, tool_name, tool_args):
        """执行工具并返回结果"""
    
    def _should_stop(self, response, step):
        """判断是否应该停止循环"""
    
    def run(self, user_input):
        """ReAct 循环主入口"""
    
    def stream_run(self, user_input):
        """流式版 ReAct 循环（逐步输出 Thought/Action/Observation）"""
```

**产出物：** `core/react_agent.py` — 核心引擎，不依赖 LangChain

---

### 第 2 天 — 工具注册系统

```
目标：建立灵活的插件式工具注册体系
```

**核心代码模块：`tools/` 目录**

```
tools/
├── __init__.py
├── registry.py          # 工具注册中心
├── base_tool.py         # 工具基类
│
├── calculator.py        # 计算器工具
├── weather.py           # 天气查询（模拟/真实 API）
├── datetime_tool.py     # 日期时间工具
├── file_ops.py          # 文件读写工具
├── web_search.py        # 网页搜索（模拟/SerpAPI）
├── code_executor.py     # 代码执行器（沙箱）
└── memory_tool.py       # 记忆存储/检索工具
```

**工具定义规范：**

```python
# base_tool.py
class BaseTool:
    name: str           # 工具名称
    description: str    # 工具描述（LLM 理解的依据）
    parameters: dict    # 参数 Schema（JSON Schema 格式）
    
    def execute(self, **kwargs) -> str:
        """执行工具逻辑"""
    
    def to_openai_tool_spec(self) -> dict:
        """转换为 OpenAI Function Calling 格式"""
    
    def to_anthropic_tool_spec(self) -> dict:
        """转换为 Anthropic Tool Use 格式"""
```

**产出物：** 5 个以上可用工具 + 统一注册机制

---

### 第 3 天 — 记忆系统

```
目标：让 Agent 具备短期 + 长期记忆能力
```

**核心代码模块：`memory/`**

```python
memory/
├── __init__.py
├── base_memory.py       # 记忆基类
├── buffer_memory.py     # 短期记忆（对话缓冲区）
├── summary_memory.py    # 摘要记忆（自动摘要历史）
├── vector_memory.py     # 向量记忆（基于 Embedding 检索）
└── hybrid_memory.py     # 混合记忆（组合多种记忆）
```

**关键设计：**

```python
class HybridMemory:
    """
    三层记忆架构：
    - Buffer:   最近 N 轮对话（上下文窗口）
    - Summary:  历史对话的压缩摘要
    - Vector:   关键信息持久化存储（向量检索）
    """
    
    def add(self, role, content):
        """添加一条记忆"""
    
    def get_context(self, query) -> str:
        """获取当前上下文（供 LLM 使用）"""
    
    def search(self, query, k=3) -> list:
        """搜索相关记忆"""
    
    def summarize(self):
        """触发摘要压缩"""
```

**产出物：** `memory/` 模块 + 多轮对话记忆测试

---

### 第 4 天 — CLI/Web 交互 + 项目集成

```
目标：构建 CLI 和 Web 两种交互方式，整体集成
```

**CLI 交互：`cli.py`**

```python
# 命令行交互模式
$ python cli.py --tools calculator,weather,datetime

Agent> 你好！我是你的 AI 工具箱助手。请问需要什么帮助？
You> 计算 23 * 45 等于多少？
Agent> 🤔 Thought: 用户需要计算乘法，使用 calculator 工具
🛠 Action: calculator(a=23, op="*", b=45)
📊 Observation: 1035
💡 Final Answer: 23 × 45 = 1035
```

**Web 交互：`app.py`（Gradio/Streamlit）**

```python
# Web 交互特点
- 聊天界面显示 Thought/Action/Observation 过程
- 工具调用可视化（哪个工具、参数、结果）
- 实时切换可用工具
- Agent 运行日志导出
```

**项目目录结构（最终）：**

```
react-agent-toolkit/
├── cli.py                      # 命令行入口
├── app.py                      # Web 入口
├── requirements.txt
├── .env.example
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── react_agent.py          # ReAct 引擎
│   └── config.py               # 配置
│
├── tools/
│   ├── __init__.py
│   ├── registry.py
│   ├── base_tool.py
│   ├── calculator.py
│   ├── weather.py
│   ├── datetime_tool.py
│   ├── file_ops.py
│   ├── web_search.py
│   ├── code_executor.py
│   └── memory_tool.py
│
├── memory/
│   ├── __init__.py
│   ├── base_memory.py
│   ├── buffer_memory.py
│   ├── summary_memory.py
│   ├── vector_memory.py
│   └── hybrid_memory.py
│
└── tests/
    ├── test_react_loop.py
    ├── test_tools.py
    └── test_memory.py
```

**产出物：** 完整 ReAct Agent 工具箱 + GitHub 仓库

---

## L2 面试考点清单

| 问题 | 考察点 | 项目对应 |
|------|--------|---------|
| 手写 ReAct 循环 | 核心原理理解 | `core/react_agent.py` |
| Function Calling 底层原理 | LLM 能力的理解 | Tool 注册 + 参数解析 |
| Agent 死循环怎么解决？ | 鲁棒性意识 | `max_steps` + `_should_stop()` |
| 工具调用参数错误怎么处理？ | 错误恢复 | 异常捕获 + 重新调用 |
| 多工具场景怎么规划？ | 推理能力 | 复杂指令的拆解 |

---

# L3 🟠 Multi-Agent 代码审查平台

> 难度：⭐⭐⭐ 进阶级  
> 目标：掌握 LangGraph 状态图 + 多 Agent 编排  
> 面试覆盖：LangGraph、状态管理、条件路由、Multi-Agent 协作模式

## 项目简介

多个 AI Agent 协同完成代码审查任务：Reviewer Agent 审查代码 → Tester Agent 生成测试 → Reporter Agent 生成报告 → 带循环修正机制。

## 详细设计路线

### 第 1-2 天 — LangGraph 基础与状态图设计

```
目标：理解 StateGraph、Node、Edge、Conditional Edge
```

**状态定义：`state.py`**

```python
from typing import TypedDict, Annotated, List, Optional
import operator

class CodeReviewState(TypedDict):
    # 输入
    code: str                         # 原始代码
    language: str                     # 编程语言
    filename: str                     # 文件名
    
    # 中间状态
    review_comments: List[dict]       # 审查意见列表
    review_score: int                 # 综合评分
    issues_found: List[dict]          # 发现的问题
    severity_levels: dict             # 问题级别统计
    
    # 测试相关
    test_cases: str                   # 生成的测试用例
    test_results: str                 # 测试执行结果
    test_coverage: float              # 测试覆盖率
    
    # 控制流
    needs_fix: bool                   # 是否需要重新审查
    fix_attempts: int                 # 已修复次数
    max_fix_attempts: int             # 最大修复次数
    
    # 输出
    final_report: str                 # 最终报告
    report_format: str                # 报告格式（markdown/json/html）
```

**图结构设计：**

```python
# graph_builder.py
from langgraph.graph import StateGraph, END

workflow = StateGraph(CodeReviewState)

# 节点
workflow.add_node("code_analyzer", analyze_code)       # 代码分析
workflow.add_node("reviewer", review_code)              # 代码审查
workflow.add_node("severity_classifier", classify)      # 严重程度分类
workflow.add_node("test_generator", generate_tests)     # 生成测试
workflow.add_node("reporter", generate_report)          # 生成报告
workflow.add_node("fix_suggester", suggest_fixes)       # 建议修复

# 边
workflow.set_entry_point("code_analyzer")
workflow.add_edge("code_analyzer", "reviewer")
workflow.add_edge("reviewer", "severity_classifier")

# 条件分支
workflow.add_conditional_edges(
    "severity_classifier",
    decide_next_step,                 # 路由函数
    {
        "generate_report": "reporter",
        "suggest_fixes": "fix_suggester",
        "generate_tests": "test_generator"
    }
)

# 循环修复
workflow.add_conditional_edges(
    "fix_suggester",
    should_review_again,
    {
        "re-review": "reviewer",      # 循环回去重新审查
        "generate_tests": "test_generator"
    }
)

workflow.add_edge("test_generator", "reporter")
workflow.add_edge("reporter", END)
```

**产出物：** `state.py` + `graph_builder.py` — 完整的 LangGraph 图定义

---

### 第 3-4 天 — Agent 节点实现

```
目标：实现每个 Agent 节点的具体逻辑
```

**节点 1：代码分析器 `nodes/analyzer.py`**

```python
def analyze_code(state: CodeReviewState) -> dict:
    """
    分析的维度：
    - 代码行数、函数数、类数
    - 复杂度估算
    - 依赖分析
    - 代码结构概览
    """
    # 1. 统计代码指标
    # 2. 识别代码结构（函数、类、导入）
    # 3. 调用 LLM 分析整体代码质量
    # 4. 返回分析结果
    return {"code_analysis": analysis_result}
```

**节点 2：审查节点 `nodes/reviewer.py`**

```python
def review_code(state: CodeReviewState) -> dict:
    """
    LLM 审查维度（每个维度一个独立 Prompt）：
    
    1. 正确性审查（Bug Detection）
       - 逻辑错误、空指针、并发问题
       - 边界条件处理
       
    2. 安全审查（Security）
       - SQL 注入、XSS、敏感信息泄露
       - 输入验证
       
    3. 性能审查（Performance）
       - 时间复杂度
       - 不必要的计算/循环
       - 资源管理
       
    4. 代码风格（Style）
       - 命名规范
       - 代码重复
       - 可读性
    """
```

**节点 3：严重程度分类 `nodes/classifier.py`**

```python
def classify(state: CodeReviewState) -> dict:
    """
    将问题按严重程度分类：
    - Critical: 会导致程序崩溃/安全漏洞
    - Major: 功能错误/性能问题
    - Minor: 代码风格/可读性
    - Suggestion: 改进建议
    
    返回统计和路由决策
    """
```

**节点 4：测试生成器 `nodes/tester.py`**

```python
def generate_tests(state: CodeReviewState) -> dict:
    """
    基于代码和审查意见生成测试：
    - 正常路径测试
    - 边界条件测试
    - 异常路径测试
    - 基于审查问题的回归测试
    
    可以选择：
    a) 只生成测试代码
    b) 生成 + 执行测试（沙箱环境）
    """
```

**节点 5：报告生成器 `nodes/reporter.py`**

```python
def generate_report(state: CodeReviewState) -> dict:
    """
    汇总所有节点的输出，生成结构化报告：
    
    ## 代码审查报告
    ### 1. 概览
    - 文件：xxx.py
    - 代码行数：150
    - 审查时间：2026-06-14
    - 综合评分：7.5/10
    
    ### 2. 问题汇总
    | 严重程度 | 数量 |
    |---------|------|
    | Critical | 1 |
    | Major | 3 |
    | Minor | 5 |
    
    ### 3. 详细问题
    - [Critical] 第 45 行：可能的内存泄漏
    - [Major] 第 78 行：缺少输入验证
    
    ### 4. 改进建议
    
    ### 5. 测试覆盖率：85%
    """
```

**产出物：** `nodes/` 目录下 5 个 Agent 节点

---

### 第 5 天 — 扩展功能

```
目标：增加润色功能，提升项目亮点
```

**功能 1：多语言支持**

```python
# 配置支持的语言
SUPPORTED_LANGUAGES = {
    "python": {"ext": ".py", "linter": "pylint"},
    "javascript": {"ext": ".js", "linter": "eslint"},
    "typescript": {"ext": ".ts", "linter": "tslint"},
    "java": {"ext": ".java", "linter": "checkstyle"},
    "go": {"ext": ".go", "linter": "golint"},
    "rust": {"ext": ".rs", "linter": "clippy"},
}
```

**功能 2：批量审查模式**

```python
# 支持整个项目目录的批量审查
def batch_review(project_path: str) -> dict:
    """扫描项目目录，按文件类型批量审查"""
```

**功能 3：CI 集成**

```python
# GitHub Actions 集成
# .github/workflows/code-review.yml
"""
name: AI Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: AI Code Review
        run: python review.py --pr ${{ github.event.number }}
"""
```

**产出物：** 扩展功能模块

---

### 第 6 天 — Web UI + 可视化

```
目标：构建可视化的 Graph 流程展示
```

**Web 界面 `app.py`：**

```python
# 核心交互流程
1. 用户粘贴代码或选择文件上传
2. 选择审查语言
3. 点击"开始审查"
4. 实时展示审查进度（Graph 节点流转动画）
5. 展示结构化报告（可折叠、可下载）
6. 支持"重新审查"（调整参数后重跑）
```

**LangGraph 可视化：**

```python
# 使用 mermaid 或 graphviz 生成图结构
from langgraph.checkpoint import MemorySaver

def visualize_workflow(app):
    """生成工作流图"""
    # 方案1: Mermaid 文本
    # 方案2: Graphviz PNG
    # 方案3: 浏览器交互式图
```

**项目目录结构（最终）：**

```
multi-agent-reviewer/
├── app.py                     # Web UI 入口
├── cli.py                     # CLI 模式入口
├── requirements.txt
├── .env.example
├── README.md
│
├── graph/
│   ├── __init__.py
│   ├── state.py              # 状态定义
│   ├── graph_builder.py      # 图构建
│   └── config.py             # 图配置
│
├── nodes/
│   ├── __init__.py
│   ├── analyzer.py           # 代码分析
│   ├── reviewer.py           # 代码审查
│   ├── classifier.py         # 严重程度分类
│   ├── tester.py             # 测试生成
│   ├── fixer.py              # 修复建议
│   └── reporter.py           # 报告生成
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # 评测指标
│   ├── benchmark.py          # 基准测试
│   └── test_suite.py         # 测试套件
│
├── integrations/
│   ├── github_actions.py     # GitHub CI 集成
│   └── gitlab_ci.py          # GitLab CI 集成
│
└── tests/
    ├── test_graph.py
    ├── test_nodes.py
    └── sample_code/          # 测试用代码
```

**产出物：** 完整 Multi-Agent 代码审查平台 + GitHub 仓库

---

## L3 面试考点清单

| 问题 | 考察点 | 项目对应 |
|------|--------|---------|
| LangGraph 和普通 LangChain Chain 的区别？ | 架构理解 | StateGraph vs Chain |
| 条件路由解决了什么问题？ | 动态流程 | Conditional Edge |
| 多个 Agent 之间怎么共享状态？ | 状态管理 | TypedDict State |
| 怎么避免 Agent 循环不终止？ | 鲁棒性 | max_fix_attempts |
| 多 Agent 相比单 Agent 有什么优势？ | 架构设计 | 分工协作模式 |

---

# L4 🔴 Agent 云服务平台

> 难度：⭐⭐⭐⭐ 高阶级  
> 目标：构建生产级 Agent 平台，涵盖 MCP 协议、多租户、评测、部署  
> 面试覆盖：系统设计、MCP、工程化、评测体系

## 项目简介

一个完整的 Agent 云服务平台：通过 MCP 协议连接多种工具，提供统一 Agent 编排、执行、监控、评测能力。支持多用户、多 Agent 实例并行运行。

## 详细设计路线

### 第 1-2 天 — MCP Server 实现

```
目标：实现 Model Context Protocol 服务端，作为 Agent 的工具层
```

**核心代码：`mcp_server/server.py`**

```python
# MCP Server 核心
class MCPServer:
    """
    MCP Server 负责：
    1. 工具注册与发现
    2. 工具调用执行
    3. 资源管理
    4. 安全沙箱
    """
    
    def __init__(self):
        self.tools = {}           # 已注册的工具
        self.resources = {}       # 可共享的资源
        self.sessions = {}        # 客户端会话
        self.auth = AuthManager() # 鉴权管理
    
    def register_tool(self, tool, auth_required=False, rate_limit=None):
        """注册 MCP 工具"""
    
    def handle_request(self, request, session_id):
        """处理 MCP 请求路由"""
    
    def validate_tool_call(self, tool_name, args):
        """工具调用的参数校验"""
    
    def execute_in_sandbox(self, func, args):
        """沙箱执行（安全隔离）"""
    
    def streaming_call(self, tool_name, args):
        """流式工具调用（适用长时间任务）"""
```

**MCP 工具示例：**

```
mcp_server/tools/
├── __init__.py
├── registry.py
├── file_system/
│   ├── read.py          # 读取文件
│   ├── write.py         # 写入文件
│   ├── list_dir.py      # 列出目录
│   └── search.py        # 文件搜索
│
├── database/
│   ├── sql_query.py     # SQL 查询
│   ├── schema.py        # 数据库 Schema
│   └── export.py        # 数据导出
│
├── web/
│   ├── fetch.py         # 网页抓取
│   ├── search.py        # 搜索
│   └── screenshot.py    # 网页截图
│
└── ai_services/
    ├── llm_chat.py      # LLM 对话
    ├── embedding.py     # Embedding
    └── summarize.py     # 摘要生成
```

**产出物：** `mcp_server/` 完整实现

---

### 第 3-4 天 — Agent 编排引擎

```
目标：构建支持多租户、并发的 Agent 执行引擎
```

**核心代码：`engine/agent_runner.py`**

```python
class AgentRunner:
    """
    Agent 执行引擎：
    - 支持多个 Agent 实例并行运行
    - 资源隔离：每个 Agent 独立上下文
    - 生命周期管理：创建 → 运行 → 暂停 → 恢复 → 销毁
    - 事件驱动：基于消息队列的通信
    """
    
    async def create_agent(self, config: AgentConfig) -> str:
        """创建 Agent 实例，返回 agent_id"""
    
    async def run_agent(self, agent_id: str, task: str):
        """运行 Agent 执行任务"""
    
    async def pause_agent(self, agent_id: str):
        """暂停 Agent（保留上下文）"""
    
    async def resume_agent(self, agent_id: str):
        """恢复 Agent 执行"""
    
    async def destroy_agent(self, agent_id: str):
        """销毁 Agent 实例"""
    
    async def get_agent_status(self, agent_id: str) -> AgentStatus:
        """获取 Agent 实时状态"""
```

**任务队列：`engine/task_queue.py`**

```python
class TaskQueue:
    """
    基于 Redis/内存的任务队列：
    - 优先级调度
    - 任务去重
    - 超时处理
    - 重试机制（指数退避）
    - 结果缓存
    """
    
    async def enqueue(self, task: Task, priority: int = 0):
        """入队"""
    
    async def dequeue(self) -> Task:
        """出队"""
    
    async def retry(self, task_id: str, delay: int):
        """延迟重试"""
    
    async def get_result(self, task_id: str) -> TaskResult:
        """获取执行结果（支持等待）"""
```

**记忆服务：`engine/memory_service.py`**

```python
class MemoryService:
    """
    统一记忆服务：
    - 支持多种记忆后端（Redis / 向量数据库 / SQLite）
    - 自动摘要压缩（减少 Token 消耗）
    - 跨会话记忆持久化
    - 记忆检索与重排序
    """
```

**产出物：** `engine/` 执行引擎

---

### 第 5-6 天 — 评测体系

```
目标：构建自动化 Agent 评测系统
```

**核心代码：`evaluation/eval_framework.py`**

```python
class EvalFramework:
    """
    Agent 评测框架：
    
    评测维度：
    1. 任务完成度（Task Success Rate）
    2. 效率（Token 消耗、延迟、步数）
    3. 质量（输出质量、幻觉率）
    4. 鲁棒性（异常处理、恢复能力）
    """
    
    def create_eval_set(self, test_cases: List[TestCase]):
        """创建评测集"""
    
    async def run_eval(self, agent, eval_set) -> EvalReport:
        """运行评测"""
    
    def compare(self, report_a, report_b) -> Comparison:
        """对比两次评测结果（A/B Test）"""
    
    def regression_check(self, baseline, current) -> bool:
        """回归检查：当前版本相比基线是否退化"""
```

**测试用例设计：`evaluation/test_cases/`**

```python
# 测试用例格式
{
    "id": "TC-001",
    "category": "tool_calling",     # 测试分类
    "difficulty": "easy",           # 难度
    "input": "帮我查一下北京的天气",
    "expected_tools": ["weather"],  # 预期调用的工具
    "expected_steps": 2,            # 预期步数
    "success_criteria": {           # 成功标准
        "contains": ["北京", "温度"],
        "tool_called": True,
        "max_steps": 5
    }
}
```

**评测报告示例：**

```
## Eval Report - Agent v1.2.3

### Overall Score: 87.3/100 📊

| Category | Score | Cases | Trend |
|----------|-------|-------|-------|
| Basic Q&A | 95% | 20/20 | ↑ +2% |
| Tool Calling | 88% | 30/34 | → 0% |
| Multi-step | 72% | 18/25 | ↓ -5% |
| Error Handling | 80% | 16/20 | ↑ +10% |
| Hallucination | 92% | 46/50 | → 0% |

### Latency: avg 1.2s (p95: 2.8s)
### Token Cost: avg 1,245 tokens/run
```

**产出物：** `evaluation/` 评测框架

---

### 第 7-8 天 — API 服务 + 多租户

```
目标：构建 FastAPI 后端，支持多用户
```

**核心代码：`api/main.py`**

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Agent Cloud Platform")

# 数据模型
class CreateAgentRequest(BaseModel):
    name: str
    tools: List[str]
    config: Optional[AgentConfig] = None

class RunTaskRequest(BaseModel):
    agent_id: str
    task: str
    stream: bool = False

# API 端点
@app.post("/api/v1/agents")
async def create_agent(req: CreateAgentRequest, user=Depends(get_current_user)):
    """创建 Agent 实例"""
    
@app.post("/api/v1/agents/{agent_id}/run")
async def run_agent(agent_id: str, req: RunTaskRequest, user=Depends(get_current_user)):
    """运行 Agent 任务"""
    
@app.get("/api/v1/agents/{agent_id}/status")
async def get_status(agent_id: str, user=Depends(get_current_user)):
    """获取 Agent 状态"""
    
@app.get("/api/v1/agents/{agent_id}/history")
async def get_history(agent_id: str, page: int = 1, user=Depends(get_current_user)):
    """获取执行历史"""
    
@app.post("/api/v1/eval/run")
async def run_evaluation(agent_id: str, eval_set: str, user=Depends(get_current_user)):
    """运行评测"""
    
@app.get("/api/v1/eval/reports/{report_id}")
async def get_report(report_id: str, user=Depends(get_current_user)):
    """获取评测报告"""
```

**认证与权限：`api/auth.py`**

```python
class AuthManager:
    """
    多租户认证：
    - JWT Token 认证
    - API Key 认证（用于自动化调用）
    - RBAC 权限控制
    - 调用频率限制
    """
```

**产出物：** `api/` 全部端点

---

### 第 9-10 天 — Docker 部署 + 项目集成

```
目标：容器化部署，项目整体集成
```

**Docker 多服务编排：**

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - chroma
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/agent_platform
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
  
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=agent_platform
      - POSTGRES_PASSWORD=password
```

**项目目录结构（最终）：**

```
agent-cloud-platform/
├── docker-compose.yml          # Docker 编排
├── Dockerfile.api              # API 服务镜像
├── Dockerfile.mcp              # MCP 服务镜像
├── requirements.txt
├── .env.example
├── README.md
│
├── api/
│   ├── main.py                 # FastAPI 入口
│   ├── auth.py                 # 认证鉴权
│   ├── models.py               # Pydantic 模型
│   ├── dependencies.py         # 依赖注入
│   └── routers/
│       ├── agents.py           # Agent 管理
│       ├── tasks.py            # 任务管理
│       └── evaluation.py       # 评测管理
│
├── engine/
│   ├── agent_runner.py         # Agent 执行器
│   ├── task_queue.py           # 任务队列
│   ├── memory_service.py       # 记忆服务
│   └── scheduler.py            # 调度器
│
├── mcp_server/
│   ├── server.py               # MCP 主服务
│   ├── protocol.py             # 协议实现
│   ├── sandbox.py              # 沙箱
│   └── tools/                  # 工具目录
│
├── evaluation/
│   ├── framework.py            # 评测框架
│   ├── metrics.py              # 指标计算
│   ├── test_cases/             # 测试用例
│   └── reports/                # 报告模板
│
├── storage/
│   ├── db.py                   # 数据库操作
│   ├── models/                 # ORM 模型
│   └── migrations/             # 数据库迁移
│
├── monitoring/
│   ├── metrics.py              # 监控指标
│   ├── logging.py              # 日志系统
│   └── tracing.py              # 链路追踪
│
├── config/
│   ├── settings.py             # 配置管理
│   └── constants.py            # 常量定义
│
└── tests/
    ├── test_api.py
    ├── test_engine.py
    ├── test_mcp.py
    └── test_evaluation.py
```

**产出物：** 完整 Agent 云服务平台 + GitHub 仓库

---

## L4 面试考点清单

| 问题 | 考察点 | 项目对应 |
|------|--------|---------|
| MCP 协议是怎么设计的？ | 协议理解 | `mcp_server/` |
| 多租户场景怎么隔离资源？ | 系统设计 | Auth + 独立上下文 |
| Agent 大规模并发怎么设计？ | 架构能力 | 任务队列 + 异步引擎 |
| 怎么评估 Agent 质量？ | 工程化意识 | 评测框架 |
| 线上 Agent 出问题了怎么排查？ | 运维能力 | 监控 + 日志 + 追踪 |
| 怎么保证工具调用的安全性？ | 安全意识 | 沙箱 + 参数校验 |

---

## 选择策略 & 面试搭配

### 时间有限（1 个月）

```
最佳组合：L1 + L2（Week 1 做完，必保）
          + L3（Week 2-3，简历亮点）
          + 简单 Docker 化（Week 4 半天）
```

### 时间充裕（2-3 个月）

```
最佳组合：L1 + L2（第 1 周，基础）
          + L3（第 2-3 周，核心）
          + L4（第 4-6 周，亮点）
          + 全部 Docker 化 + CI/CD（第 7-8 周）
          + 部署上线（第 9-10 周）
```

### 面试岗位匹配

| 岗位类型 | 推荐项目 | 面试侧重点 |
|---------|---------|-----------|
| Agent 应用开发 | L1 + L2 + L3 | 项目完整性、框架使用 |
| Agent 平台开发 | L4 | 系统设计、架构能力 |
| 通用 AI 开发 | L1 + L3 | RAG + Multi-Agent 编排 |

---

## 项目贡献指南

每个项目需要包含以下文件才有简历投递资格：

```
□ README.md          — 项目简介 + 效果图 + 架构说明
□ requirements.txt   — 依赖管理
□ .env.example       — 配置模板
□ 可运行的入口       — app.py / cli.py
□ 核心代码模块       — 按功能分层
□ 测试代码           — 至少基本测试
□ 项目结构清晰       — 命名规范、类型注解
```

**README 必须包含：**

```markdown
1. 项目名称 + 一句话简介
2. 技术栈（图标装饰更佳）
3. Demo 截图或 GIF
4. 快速开始（clone → install → run 三步）
5. 项目架构图（Mermaid）
6. 核心功能列表
7. 效果数据（如果有评测）
8. 学到什么（3-5 个技术点）
9. 后续计划（可选）
10. 联系方式
```

---

## 写在最后

**这四个项目之间是递进关系：**

- **L1** 让你理解 RAG 和文档处理
- **L2** 让你深入 Agent 核心循环
- **L3** 让你掌握多 Agent 编排
- **L4** 让你具备生产级工程视野

能做完 L1+L2，你已经具备面试基本盘。  
能做完 L3，你有简历亮点。  
能做完 L4，你就是应届生里的"架构师"级别。

**不需要全部做完才投简历。** 每做完一个就更新 GitHub，更新简历，继续投递。
