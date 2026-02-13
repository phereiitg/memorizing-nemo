# Memorizing-Nemo 🐠

**A Multi-Agent Long-Form Memory System for LLMs**

**Memorizing-Nemo** is an advanced 5-layer cognitive architecture designed to give Large Language Models (LLMs) persistent, self-managing long-term memory. Unlike simple RAG (Retrieval Augmented Generation) wrappers, Nemo uses a biological-inspired **Heat Decay System** to mimic human memory: information is retained based on relevance and recency, and fades over time if unused.

It enables an AI system to remember facts, preferences, and constraints across **1000+ conversation turns** without exploding the context window.

---

## ✨ Key Features

* 🧠 **True Long-Term Memory** beyond prompt context limits
* 🔥 **Heat / Decay-based Memory Retention** (biologically inspired)
* 🧩 **5-Layer Agentic Architecture** with clear separation of concerns
* ⚡ **Low-latency synchronous path** for fast responses
* 🗄️ **Hybrid Storage System** (RAM + Vector DB + SQLite)
* 🔄 **Background Memory Maintenance** (conflict resolution & decay)

---

## ⚡ Quick Start

### 1️⃣ Prerequisites

* **Python 3.10 or 3.11 (Recommended)**

  * ⚠️ *Python 3.13 is currently experimental for some dependencies (notably ChromaDB)*
* A **Google Gemini API Key**
* Git installed

---

### 2️⃣ Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/yourusername/memorizing-nemo.git
cd memorizing-nemo
```

Create and activate a virtual environment (**recommended**):

```bash
# Create virtual environment using Python 3.10
py -3.10 -m venv venv

# Activate (Windows)
venv\\Scripts\\activate

# Activate (Linux / macOS)
source venv/bin/activate
```

Install dependencies:

```bash
pip install google-generativeai chromadb python-dotenv
```

---

### 3️⃣ Configuration

**🔐 Security Notice:** Never hardcode API keys inside source code.

1. Create a `.env` file in the project root
2. Add your Google Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

The `.env` file is automatically loaded at runtime using `python-dotenv`.

---

### 4️⃣ Running the System

Start an interactive CLI chat session:

```bash
python main.py
```

---

## ⚠️ Important: First Run Warning

On the **first execution only**, the system will automatically download a local embedding model:

* **Model:** `all-MiniLM-L6-v2`
* **Purpose:** Semantic embeddings for the vector database (ChromaDB)

📌 This behavior is **expected and required**.

* ⏱️ **Duration:** ~1–2 minutes (depends on internet speed)
* ⛔ **Do not interrupt the process**
* ✅ This happens **only once**

---

## 🏗 System Architecture Overview

Memorizing-Nemo is designed as a **5-Layer Agentic Pipeline**, clearly separating:

* **Synchronous (Critical Path)** → Used during response generation
* **Asynchronous (Background Tasks)** → Used for memory extraction, decay, and maintenance

---

## 🏛️ The 5 Cognitive Layers

| Layer  | Agent Name     | Execution Type | Responsibility                                                  |
| ------ | -------------- | -------------- | --------------------------------------------------------------- |
| **L1** | Sentinel       | Asynchronous   | Extracts facts, preferences, and constraints from conversations |
| **L2** | Memory Store   | Hybrid         | Manages Hot / Warm / Cold memory tiers                          |
| **L3** | Oracle         | Synchronous    | Retrieves relevant memories before generation                   |
| **L4** | Response Agent | Synchronous    | Calls Gemini LLM and generates final response                   |
| **L5** | Curator        | Asynchronous   | Resolves conflicts, applies decay, and manages lifecycle        |

---

## 🔁 End-to-End System Flowchart

```text
┌──────────────┐
│   User Input │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Mnemosyne Engine     │
│ (main.py / engine)   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Oracle (L3)          │◄───────────────┐
│ Retrieve Memories    │                │
└──────┬───────────────┘                │
       │                                │
       ▼                                │
┌──────────────────────┐                │
│ Memory Store (L2)    │───────────────►│
│ Hot / Warm / Cold    │                │
└──────┬───────────────┘                │
       │                                │
       ▼                                │
┌──────────────────────┐                │
│ Response Agent (L4)  │                │
│ Gemini LLM           │                │
└──────┬───────────────┘                │
       │                                │
       ▼                                │
┌──────────────────────┐                │
│   User Response      │                │
└──────────────────────┘                │
                                        │
        ───── Async Background ─────    │
                                        │
┌──────────────────────┐                │
│ Sentinel (L1)        │                │
│ Extract Memories     │                │
└──────┬───────────────┘                │
       │                                │
       ▼                                │
┌──────────────────────┐                │
│ Curator (L5)         │────────────────┘
│ Validate & Decay     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Memory Store (L2)    │
│ Commit / Evict       │
└──────────────────────┘
```

---

## 🧠 Deep Dive: Memory Store (Layer 2)

### 🔥 Hot Tier (RAM)

* Implemented using a deque
* Holds short-term conversational context
* **O(1)** access time
* Cleared frequently to control context size

### 🌡️ Warm Tier (Vector Store – ChromaDB)

* Semantic associative memory
* Uses sentence embeddings (`all-MiniLM-L6-v2`)
* Retrieves memories based on relevance + heat score

### ❄️ Cold Tier (SQLite)

* Long-term persistent storage
* Stores raw interactions and structured logs
* Supports auditing, replay, and analysis

---

## 📂 Repository Structure

```text
memorizing-nemo/
├── main.py                # Entry point (CLI)
├── .env                   # API keys (excluded from git)
├── mnemosyne_memories.db  # SQLite DB (auto-generated)
├── mnemosyne_chroma/      # Vector DB files (auto-generated)
├── mnemosyne_logs.jsonl   # Interaction logs (auto-generated)
├── requirements.txt       # Dependencies
└── mnemosyne/
    ├── engine.py          # Main orchestration loop
    ├── models.py          # Data models
    ├── sentinel.py        # L1: Memory extraction agent
    ├── memory_store.py    # L2: Storage controller
    ├── oracle.py          # L3: Retrieval agent
    └── curator.py         # L5: Memory lifecycle manager
```

---

## 🎛️ Customization

* Change the LLM model in `main.py`
* Modify memory extraction logic in `mnemosyne/sentinel.py`
* Tune decay rates and retention thresholds in `curator.py`

The system defaults to **`gemini-1.5-flash`** for speed and efficiency.

---

## 🚀 Roadmap (Optional)

* Multi-user memory isolation
* Memory visualization dashboard
* Pluggable LLM backends
* Distributed vector storage

---

## 📜 License

MIT License

---

**Memorizing-Nemo** — Teaching machines how to remember 🐠🧠
