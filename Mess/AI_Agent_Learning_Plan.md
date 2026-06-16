# 🧠 AI Agent 岗位 1 个月极限冲刺计划

> 适用对象：两年制研一学生 · 仅剩 1 个月准备 · 边投边面  
> 核心策略：**第 1 周做出简历项目 → 第 2 周开始投递 → 边面边补**  
> 原则：只做最高 ROI 的事，砍掉所有低效学习

---

## 一、残酷的时间分配

```
Week 1 ──── Week 2 ──── Week 3 ──── Week 4
   |           |           |           |
  🛠 做项目    🎯 投递     🧪 深挖      🔥 冲刺
  (出简历项目) (边面边做)  (面经复盘)   (查漏补缺)
```

| 周次 | 核心目标 | 投入时间预估 |
|------|---------|-------------|
| **Week 1** | 做出 2 个可上简历的项目 | 每天 8-10h |
| **Week 2** | 开始投递 + 第 3 个项目 + 刷题 | 每天 6-8h |
| **Week 3** | 面试复盘 + 项目 Docker 化 + 算法强化 | 每天 6-8h |
| **Week 4** | 查漏补缺 + 面试密集期 + 深度准备 | 每天 6-8h |

---

## 二、Week 1：疯狂做项目（这是你唯一的救命稻草）

> ⚠️ 警告：没有项目，简历就是废纸。Week 1 不睡觉也要产出两个项目。

### 📅 每日时间表

```
07:00-08:00  LeetCode 1-2 道（保持手感）
08:00-12:00  项目 coding（上午大块时间）
12:00-13:00  吃饭 + 休息
13:00-18:00  项目 coding（下午大块时间）
18:00-19:00  吃饭 + 休息
19:00-22:00  继续 coding / 写 README / 整理 GitHub
22:00-23:00  复盘 + 看面试题（当催眠）
```

### 🏗 项目 1：RAG 智能问答系统（Day 1-4）

> **这是你简历上最重要的项目**，80% 的 Agent 岗面试都会问 RAG。

#### 功能规格（最小可用版）
```
用户上传 PDF/TXT → 自动分块 → 向量化存储 → 自然语言提问 → 基于文档回答
```

#### 技术栈
```
Python + LangChain + ChromaDB + OpenAI API + Streamlit
```

#### Day 1 就能跑起来的核心代码

**Step 1: 文档加载与分块**（save as `rag/loader.py`）

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()

def split_documents(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
```

**Step 2: 向量存储**（save as `rag/vector_store.py`）

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def create_vectorstore(docs, persist_dir="./chroma_db"):
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore

def load_vectorstore(persist_dir="./chroma_db"):
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
```

**Step 3: 检索问答链**（save as `rag/qa_chain.py`）

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

def ask(chain, question: str):
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }
```

**Step 4: Streamlit UI**（save as `app.py`）

```python
import streamlit as st
from rag.loader import load_document, split_documents
from rag.vector_store import create_vectorstore, load_vectorstore
from rag.qa_chain import create_qa_chain, ask
import tempfile
import os

st.set_page_config(page_title="RAG 智能知识库", layout="wide")
st.title("📚 RAG 智能知识库系统")

# 上传文档
uploaded_file = st.file_uploader("上传 PDF 或 TXT 文件", type=["pdf", "txt"])
if uploaded_file and "vectorstore" not in st.session_state:
    with st.spinner("正在处理文档..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
            f.write(uploaded_file.read())
            file_path = f.name
        
        docs = load_document(file_path)
        chunks = split_documents(docs)
        vectorstore = create_vectorstore(chunks)
        st.session_state.vectorstore = vectorstore
        st.session_state.chain = create_qa_chain(vectorstore)
        st.success(f"✅ 文档处理完成！共 {len(chunks)} 个片段")
        os.unlink(file_path)

# 问答
if "chain" in st.session_state:
    st.subheader("💬 基于文档提问")
    question = st.text_input("输入你的问题：")
    if question:
        with st.spinner("思考中..."):
            result = ask(st.session_state.chain, question)
        st.markdown(f"**答案：** {result['answer']}")
        with st.expander("📎 查看来源"):
            for src in result["sources"]:
                st.json(src)
```

**Step 5: 运行**

```bash
pip install langchain langchain-openai langchain-chroma chromadb streamlit pypdf
export OPENAI_API_KEY="sk-xxx"
streamlit run app.py
```

> **2 小时内就能跑起来**。关键不是代码多复杂，而是你能讲清楚原理。

#### 必须能回答的面试问题
```
1. 为什么选择 Chunk Size 500/100？不同 chunk 策略有什么影响？
2. 只用一个 Embedding 检索有什么缺陷？Hybrid Search 怎么改进？
3. 怎么评估 RAG 的检索质量？
4. 如果用户问的问题文档里没有，怎么处理？
5. 怎么优化检索速度？
```

---

### 🏗 项目 2：Tool Calling Agent（Day 5-7）

> **这是你的第二个简历项目**，展示你对 Agent 核心机制的理解。

#### 功能规格
```
一个能调用外部工具的 Agent，支持：
  - 天气查询（模拟/真实 API）
  - 计算器功能
  - 文件读写操作
  - 网页搜索摘要
```

#### 核心实现：手写 ReAct 循环（save as `react_agent.py`）

```python
import json
import openai
from typing import Dict, Any

class ReActAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.tools = {}
    
    def register_tool(self, name: str, func, description: str, parameters: dict):
        """注册工具"""
        self.tools[name] = {
            "func": func,
            "spec": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            }
        }
    
    def run(self, user_input: str, max_steps: int = 10) -> str:
        """ReAct 循环主入口"""
        messages = [
            {"role": "system", "content": "你是一个能调用工具的助手。按以下格式思考：\n"
                                          "Thought: 分析用户需求\n"
                                          "Action: 选择工具\n"
                                          "Observation: 工具返回结果\n"
                                          "Final Answer: 最终回答"},
            {"role": "user", "content": user_input}
        ]
        
        tool_specs = [t["spec"] for t in self.tools.values()]
        
        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_specs if tool_specs else None,
                tool_choice="auto" if tool_specs else None
            )
            
            msg = response.choices[0].message
            
            # 没有工具调用 → 直接返回
            if not msg.tool_calls:
                return msg.content
            
            # 有工具调用 → 执行并继续循环
            messages.append(msg)
            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                
                print(f"  🛠 Step {step+1}: {func_name}({func_args})")
                
                if func_name in self.tools:
                    result = self.tools[func_name]["func"](**func_args)
                else:
                    result = f"Error: 未知工具 {func_name}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result)
                })
        
        return "已达到最大步数限制"
```

**使用示例**（save as `demo.py`）

```python
from react_agent import ReActAgent
import math

agent = ReActAgent(api_key="sk-xxx")

# 注册计算器工具
agent.register_tool(
    name="calculator",
    func=lambda a, op, b: 
        a + b if op == "+" else
        a - b if op == "-" else
        a * b if op == "*" else
        a / b if op == "/" else "未知运算符",
    description="执行四则运算",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "op": {"type": "string", "enum": ["+", "-", "*", "/"]},
            "b": {"type": "number"}
        },
        "required": ["a", "op", "b"]
    }
)

# 注册获取当前时间工具
from datetime import datetime
agent.register_tool(
    name="get_current_time",
    func=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    description="获取当前日期和时间",
    parameters={"type": "object", "properties": {}}
)

# 运行
print(agent.run("现在几点了？"))
print(agent.run("计算 (15 + 27) * 3 等于多少"))
```

#### 面试必问题
```
1. 手写 ReAct 循环（上面这个就是答案）
2. Tool Calling 的底层原理是什么？LLM 怎么知道调用哪个工具？
3. 如果工具返回错误，Agent 怎么处理？
4. 怎么防止 Agent 陷入死循环？
5. 多工具调用时怎么决定调用顺序？
```

---

## 三、Week 2：投递 + 第 3 个项目 + 刷题

### 🎯 立即开始投递

**第一优先级（内推 > 官网 > 招聘平台）**

| 渠道 | 操作 | 说明 |
|------|------|------|
| 找师兄师姐内推 | **最优路径** | 直接推简历到负责人 |
| 牛客网 | 找内推帖 + 面经 | 搜索"AI Agent 内推" |
| 大厂校招官网 | 字节/阿里/腾讯/百度/美团 | 直接投递 |
| Boss直聘 | 主动打招呼 | 写一段简介模板 |
| 实习僧 | 批量投递 | 不放过机会 |

**简历投递策略**
```
- 每天投 5-10 家
- 先投中小厂练手面试
- 大厂放后面（面挂了会有冷冻期）
- 每次面试后复盘，补弱项
```

### 🏗 项目 3：LangGraph Multi-Agent 系统（Week 2 主力）

> 第三个简历项目。展示你对 Multi-Agent 编排的理解。

#### 功能：AI 代码审查系统（save as `code_review_agent.py`）

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import operator

# 定义状态
class ReviewState(TypedDict):
    code: str                    # 待审查的代码
    review_result: str          # 审查结果
    test_cases: str             # 测试用例
    final_report: str           # 最终报告

# 定义节点函数
def reviewer(state: ReviewState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""审查以下 Python 代码，检查：
1. 潜在的 Bug
2. 代码风格问题
3. 性能问题
4. 安全漏洞

代码：
```python
{state['code']}
```
"""
    result = llm.invoke(prompt)
    return {"review_result": result.content}

def test_generator(state: ReviewState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = f"""根据以下代码生成 pytest 测试用例：
```python
{state['code']}
```
审查意见：{state['review_result']}
"""
    result = llm.invoke(prompt)
    return {"test_cases": result.content}

def reporter(state: ReviewState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""汇总以下信息为一份完整的代码审查报告：
    
## 原始代码
```python
{state['code']}
```

## 审查意见
{state['review_result']}

## 测试用例
{state['test_cases']}

请按以下格式输出：
- 问题总结
- 严重程度分级
- 修改建议
- 测试覆盖率评估
"""
    result = llm.invoke(prompt)
    return {"final_report": result.content}

def should_continue(state: ReviewState):
    """条件路由：检查审查结果是否需要重新审查"""
    if "严重问题" in state.get("review_result", ""):
        return "需修复后重新审查"
    return "生成最终报告"

# 构建图
workflow = StateGraph(ReviewState)

workflow.add_node("reviewer", reviewer)
workflow.add_node("test_generator", test_generator)
workflow.add_node("reporter", reporter)

workflow.set_entry_point("reviewer")
workflow.add_edge("reviewer", "test_generator")
workflow.add_conditional_edges(
    "test_generator",
    should_continue,
    {
        "需修复后重新审查": "reviewer",  # 循环
        "生成最终报告": "reporter"
    }
)
workflow.add_edge("reporter", END)

app = workflow.compile()

# 运行
result = app.invoke({"code": open("my_code.py").read()})
print(result["final_report"])
```

#### 面试必问题
```
1. LangGraph 和 LangChain 的区别是什么？
2. StateGraph 的节点状态是怎么传递的？
3. 条件路由（Conditional Edge）解决了什么问题？
4. 多个 Agent 之间怎么通信？
5. 怎么防止 Agent 陷入无限循环？
```

---

## 四、Week 3：面试复盘 + 项目 Docker 化 + 算法强化

### 🐳 项目 Docker 化

把 Week 1 的 RAG 项目 Docker 化（加分项，半天搞定）

**Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

**docker-compose.yml**

```yaml
version: '3.8'
services:
  rag-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db
```

### 📝 每日新增：面试复盘模板

每次面试后立即记录：

```
## 面试复盘 - {公司名} - {日期}

### 被问到的问题
1. 
2. 
3. 

### 没答上来的
- 

### 下次要补的
- 

### 面试感受
- 
```

### 💪 算法刷题策略（有时间就刷）

**只刷高频，不刷难题**

```
优先级：
  1️⃣ LeetCode Hot 100  —— 大厂最爱
  2️⃣ 剑指 Offer 专项  —— 经典题
  3️⃣ 按 tag 刷：
      - 数组/字符串
      - 哈希表
      - 二叉树/DFS/BFS
      - 动态规划（入门级）
  
每日目标：2-3 道（Easy/Medium）
面试前总目标：刷够 80-100 道高质量题
```

---

## 五、Week 4：冲刺 + 补漏

### 📋 查漏补缺清单

```
□ 简历上至少有 2 个项目（RAG + Multi-Agent）
□ GitHub 仓库 README 完整
□ 每个项目都能讲 5 分钟（原理 + 难点 + 改进方向）
□ 算法刷够 80 题
□ 看过 20 篇 AI Agent 面经
□ 投递了至少 30 家公司
□ 准备好自我介绍（1 分钟/3 分钟两个版本）
```

### 🔥 Agent 面试高频题突击

**第一类：基础概念（必问）**
```
1. 什么是 ReAct 模式？和 Plan-and-Solve 有什么区别？
   → 回答：ReAct = Reasoning + Acting 交替循环，每一步先推理再行动；
     Plan-and-Solve 是先规划全部步骤再逐步执行。
     
2. LLM Tool Calling 的原理是什么？
   → 回答：本质是 LLM 输出结构化 JSON（函数名+参数），
     代码层解析后调用对应函数，函数结果再送回 LLM。
     关键：Tool Schema 的定义直接影响 LLM 选择的准确率。

3. RAG 的完整流程是什么？有哪些优化点？
   → 回答：文档 → Chunk → Embedding → 存储 → 检索 → 重排序 → LLM 生成
     优化点：Chunk 策略、Hybrid Search、Query Rewrite、Re-ranking

4. 怎么评估 Agent 的质量？
   → 回答：任务完成率、Token 消耗、延迟、成功率；
     LLM-as-Judge 做自动化评测
```

**第二类：场景设计（拉开差距）**
```
5. 设计一个客服 Agent 系统
6. 如何让 Agent 拥有长期记忆？
7. 多 Agent 之间怎么协作？有什么挑战？
8. Agent 出现幻觉怎么处理？
```

**第三类：手写代码**
```
9. 手写一个 ReAct 循环（答：看我 Week 1 项目 2 的代码）
10. 手写文档 Chunking 逻辑
11. 手写简单的 Embedding 相似度检索
```

### 🎬 面试当天 checklist

```
□ 准备好安静的环境
□ 耳机 + 摄像头正常
□ 屏幕共享时只开必要窗口
□ 自我介绍流畅（1min / 3min）
□ 项目准备好 Demo / 能直接打开 GitHub 展示代码
□ 准备好问面试官的问题：
   - "团队目前在 Agent 方向主要做什么？"
   - "这个岗位的日常工作是偏研究还是偏工程？"
   - "团队用什么 Agent 框架/技术栈？"
```

---

## 六、最高 ROI 行动清单（这一个月照着做）

### Week 1 每天必须完成的项目里程碑

| 天 | 项目进度 | 代码产出 |
|----|---------|---------|
| D1 | RAG Step 1-2：文档加载 + 分块 + 向量存储 | `rag/loader.py` `rag/vector_store.py` |
| D2 | RAG Step 3-4：QA Chain + Streamlit UI | `rag/qa_chain.py` `app.py` |
| D3 | RAG 调通 + 测试 + 写 README | 完整可运行 + GitHub 仓库 |
| D4 | RAG README 完善 + 准备面试问题 | 博客/笔记 |
| D5 | ReAct Agent：核心循环 + 注册工具 | `react_agent.py` |
| D6 | ReAct Agent demo + 多工具测试 | `demo.py` |
| D7 | ReAct Agent README + 两个项目都上 GitHub | GitHub 双项目 |

### 面试准备的 20% 关键知识点

```
不用看（低ROI）：
  ✗ LLM 训练细节（SFT/DPO/RLHF）—— 算法岗才需要
  ✗ 模型微调 —— 同上
  ✗ Kubernetes 深度使用 —— 有时间再看
  ✗ 各种论文精读 —— 没时间

必须吃透（高ROI）：
  ✓ ReAct 循环原理 —— 面试手写
  ✓ Tool Calling 原理 —— 核心考察点
  ✓ RAG 全流程 —— 简历重点项目
  ✓ LangGraph 状态图 —— Multi-Agent 核心
  ✓ Agent 评测与优化 —— 拉开差距
  ✓ 你的两个项目代码 —— 面试必深挖
```

---

## 七、心态策略

### ⚡ 这一个月你的身份是"创业 CEO"

```
- 你的产品 = 你的简历和 GitHub 项目
- 你的融资 = 面试机会
- 你的 PMF = 面试通过率
- 你的迭代 = 每次面试后复盘改进
```

### 🎯 目标不是"懂 AI Agent"，目标是"拿到面试通过"

- 不需要懂所有东西
- 只需要懂 **你简历上写的** 和 **面试问到的**
- 面试问到不会的 → 记下来 → 下次就会了

### 💬 面试不会怎么办？

```
面试官："你对 XXX 了解吗？"

❌ "不了解，没学过。"（直接暴露短板）

✅ "这块我目前了解得还比较浅，我的理解是……（说你知道的部分），
    学习过程中我主要聚焦在 YYY 上，XXX 是我下一步计划深入的方向。"
    （诚实 + 展示学习能力）

如果完全没听过：
✅ "这块我还没接触过，能请您展开说一下吗？我记一下回去学习。"
    （把不会变成学习机会，面试官通常愿意讲）
```

---

## 八、写在最后

**1 个月，够不够？不够也得够。**

你现在的优势：
- ✅ 学过编程基础，不是从零开始
- ✅ AI Agent 岗位还在爆发期，需求大
- ✅ 大厂对实习生的期望是"有潜力"不是"全栈大师"

唯一需要做的：**Week 1 把两个项目做出来放到 GitHub 上，Week 2 开始疯狂投递。**

前 7 天是最痛苦的，每天 10h+ coding，但过了这 7 天你就有东西在简历上了。
后面的每一天，你都在变得更强。

**开始做吧，不是明天，是现在。**
