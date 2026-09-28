## Learning LangChain: From Raw API Calls to a Working AI Agent

A 15-article, hands-on series that teaches you to build applications on top of large language models (LLMs) with LangChain. You start below the framework, calling OpenAI, Anthropic and Gemini directly, then learn LangChain one piece at a time, and finish by assembling a complete, tested AI support assistant.

Every article follows the same pattern: the theory first, then a full implementation you can run. Every code block was run before it was written down, and the printed outputs in the articles are the real outputs.

## What is LangChain?

A large language model on its own is a text-in, text-out function. You send it a list of messages, and it sends back a reply. That is powerful, but it leaves a lot missing if you want to build a real product:

- **It has no memory.** Each API call starts from nothing. If you want a conversation, you have to resend the whole history every time.
- **It doesn't know your data.** A model knows what it saw during training, not your company's documentation, your customers, or anything that happened after its training cutoff.
- **It can't take actions.** It can describe how to check an order status, but it can't actually look one up.
- **Every provider's API is different.** OpenAI, Anthropic and Google each use different request formats, parameter names and response shapes.

**LangChain** is an open-source Python framework that fills those gaps. It gives you:

- **One interface for every model provider**, so switching from Claude to GPT to Gemini is a one-line change.
- **Building blocks** for the things every LLM application needs: prompt templates, structured output, conversation memory, document loading, embeddings, vector search, and tools.
- **A composition pattern (LCEL)** for snapping those blocks together into pipelines with the `|` operator.
- **Agents** (built on its sister project, LangGraph) that let a model decide which tools to call, run them, and keep going until it has an answer.
- **Observability** through LangSmith, so you can see exactly what happened inside every run.

This series teaches LangChain the way it makes the most sense: by first doing everything by hand with the raw provider SDKs, so that when LangChain automates something, you already know what it is doing for you.

## What you will build

The whole series follows one continuous project: an AI support assistant for **Tidewave**, a fictional project-management SaaS with subscription billing. Each article adds one capability to it, and nothing is thrown away between articles.

By the end, the assistant can:

- **Answer questions from Tidewave's own documentation** using retrieval-augmented generation (RAG), instead of guessing from what the model already knows.
- **Look up a specific customer's account** by calling a real Python function as a tool.
- **Decide on its own** which of those two capabilities a question needs, or whether it needs neither.
- **Remember the conversation** across turns, with long histories summarized automatically.
- **Triage every incoming message** into a category and priority with structured output, and flag urgent billing issues for a human.
- **Be traced end to end** in LangSmith, so every model call and tool call is visible.
- **Be tested** with a pytest suite that runs without any API keys.

The final project lives in [`tidewave_assistant/`](tidewave_assistant/), and [Article 15](article-15-capstone.html) walks through it file by file.

## What you will be able to do after finishing this repo

- Call OpenAI, Anthropic and Gemini models directly with their own SDKs, and understand what every parameter does.
- Write provider-agnostic LLM code, so your application isn't locked to one vendor.
- Get reliable, validated structured data (a Pydantic object) out of a model instead of free text.
- Compose prompts, models and parsers into pipelines with LCEL.
- Build a full RAG pipeline: load documents, split them, embed them, store them in a vector database, and retrieve the right chunks for a question.
- Turn Python functions into tools, and build agents that use them with `create_agent`.
- Add memory, limits and summarization to an agent with checkpointers and middleware.
- Trace, debug and test an LLM application the way you would in production.
- Recognize outdated LangChain tutorials, and know which older APIs have been replaced (a lot of material online still uses them).

## Who this repo is for

- **Python developers** who want to start building with LLMs and keep hearing about LangChain.
- **Data scientists and ML engineers** moving into generative AI and AI engineering roles.
- **Students and self-learners** who want one structured path instead of piecing together scattered tutorials.
- **Anyone who has tried LangChain before and found it confusing.** This series explains why each abstraction exists before using it.

This repo is **not** a deep dive into how LLMs are trained, and it doesn't require any machine learning background. Article 1 covers the concepts you need about how models work.

## Prerequisites

### Knowledge

- **Python fundamentals**: variables, functions, lists and dictionaries, loops, `if` statements, importing modules, and basic classes. If you can write a small script that reads a file and processes its lines, you are ready.
- **Type hints and Pydantic (helpful, not required)**: you will see code like `def f(text: str) -> str:` and `class Ticket(BaseModel):`. The articles explain what you need as it comes up.
- **JSON**: knowing what a JSON object looks like is enough.
- **No machine learning or math background is needed.**

### Tools you should be comfortable with

- **VS Code (or another code editor)**: opening a folder, creating and editing `.py` files, and using the built-in terminal (`View > Terminal`). Installing the **Python** extension from Microsoft is recommended, so VS Code can find your virtual environment.
- **The terminal**: moving between folders with `cd`, listing files with `ls` (macOS/Linux) or `dir` (Windows), and running commands such as `python main.py` and `pip install`. On Windows, PowerShell or the VS Code terminal both work.
- **Virtual environments**: creating one with `python -m venv` and activating it. The setup steps below show the exact commands.
- **Git basics (optional)**: `git clone` to download this repo. You can also download it as a ZIP from GitHub instead.

### Software to install

| What | Version | Why |
|---|---|---|
| Python | 3.10 or newer (3.11+ recommended) | LangChain 1.x requires Python 3.10+ |
| VS Code | Any recent version | Editing and running the code |
| Git | Any recent version | Cloning the repo (optional) |

### Accounts and API keys

The code calls real model providers, so you need API keys. Keep them in a `.env` file and never commit that file to GitHub.

| Service | Needed for | Required? |
|---|---|---|
| [Anthropic](https://console.anthropic.com) | Chat model (Claude), the series default | Yes |
| [OpenAI](https://platform.openai.com) | Chat model (GPT) and embeddings for RAG | Yes |
| [Google AI Studio](https://aistudio.google.com) | Gemini, used in Article 3 and as an alternative | Optional |
| [LangSmith](https://smith.langchain.com) | Tracing and observability, Article 14 | Optional, has a free tier |

Both Anthropic and OpenAI bill per use, so you will need to add a small amount of credit to each account. Running every example in the series makes a modest number of calls. The capstone's test suite runs with no keys at all.

## The articles

Read them in order. Each one builds directly on the one before it.

| # | Article | What it covers | Read the article |
|---|---|---|---|
| 1 | Generative AI basics, and where LangChain actually fits | How LLMs work, what they can't do on their own, and the LangChain ecosystem | [Read Article 1](ADD_LINK_HERE) |
| **Part 1** | **Below the framework** | | |
| 2 | Your first LLM calls: OpenAI and Anthropic, no framework yet | Raw SDK calls, message lists, streaming, and a chat loop by hand | [Read Article 2](ADD_LINK_HERE) |
| 3 | The multi-provider landscape: Gemini, other providers, and a unified interface | Adding Gemini, comparing providers, and LiteLLM | [Read Article 3](ADD_LINK_HERE) |
| 4 | Tuning generation: temperature, top-p, and the other parameters | Sampling parameters, reasoning effort, and what each provider accepts | [Read Article 4](ADD_LINK_HERE) |
| **Part 2** | **LangChain fundamentals** | | |
| 5 | Setup, and talking to models through LangChain's unified interface | `init_chat_model`, the shared model interface, and the project's `get_model()` helper | [Read Article 5](ADD_LINK_HERE) |
| 6 | Prompt templates and structured output | `ChatPromptTemplate`, `MessagesPlaceholder`, and `with_structured_output` | [Read Article 6](ADD_LINK_HERE) |
| 7 | LCEL: the pipe-operator pattern used to compose everything | `Runnable`, chains, `RunnableParallel`, `.batch()` and `.stream()` | [Read Article 7](ADD_LINK_HERE) |
| 8 | Conversation memory | Session history, why the old memory classes are gone, and `trim_messages` | [Read Article 8](ADD_LINK_HERE) |
| **Part 3** | **Retrieval (RAG)** | | |
| 9 | Document loaders and text splitting | `Document`, loading files and PDFs, and `RecursiveCharacterTextSplitter` | [Read Article 9](ADD_LINK_HERE) |
| 10 | Embeddings and vector stores | What embeddings are, Chroma, and a distance gotcha found by running the code | [Read Article 10](ADD_LINK_HERE) |
| 11 | Retrievers and a full RAG pipeline | Retrievers, formatting context, and the complete RAG chain | [Read Article 11](ADD_LINK_HERE) |
| **Part 4** | **Tools, agents and production** | | |
| 12 | Tools and tool calling | `@tool`, `bind_tools`, and the tool-calling loop written by hand | [Read Article 12](ADD_LINK_HERE) |
| 13 | Agents with `create_agent` | The agent loop automated, checkpointer memory, call limits, and middleware | [Read Article 13](ADD_LINK_HERE) |
| 14 | Observability with LangSmith | Tracing, `@traceable`, and keeping sensitive data out of traces | [Read Article 14](ADD_LINK_HERE) |
| 15 | Capstone: the whole assistant, end to end | Every piece assembled into one tested project | [Read Article 15](ADD_LINK_HERE) |

Each article is available as a styled HTML page (best for reading) and as a Markdown file in this repo. The **Read the article** column links to the published version of each one. To read the HTML files directly on GitHub, enable **GitHub Pages** for this repo (`Settings > Pages`, deploy from the `main` branch), or download the repo and open the `.html` files in your browser.

## Getting started

**1. Get the code**

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
```

**2. Create and activate a virtual environment**

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your terminal prompt should now start with `(.venv)`. In VS Code, open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`), choose **Python: Select Interpreter**, and pick the `.venv` one.

**3. Install the dependencies**

```bash
pip install -r tidewave_assistant/requirements.txt
```

The early articles use a few extra packages (the raw `openai`, `anthropic` and `google-genai` SDKs, and `litellm`). Each article lists what it needs at the top, so install those as you reach them.

**4. Add your API keys**

Copy the example file and fill in your own keys:

```bash
cp tidewave_assistant/.env.example tidewave_assistant/.env
```

**5. Start reading at Article 1**

Type the code out as you go rather than copying it. When you reach Article 15, run the finished assistant:

```bash
cd tidewave_assistant
pytest -q          # runs the test suite, no API keys needed
python main.py     # starts the assistant, needs your keys
```

## What makes this series different

- **One project, start to finish.** Tidewave's assistant grows by one capability per article, so you see how the pieces fit together instead of learning each one in isolation.
- **Verified, not just written.** Every code block was run, and every printed output shown is the real output. Checking the code this way caught real bugs, which the articles explain instead of hiding.
- **Current as of September 2026.** The series uses LangChain 1.x, `create_agent`, and current models (Claude Sonnet 5, GPT-5.5, Gemini 3.8 Flash). It points out where older tutorials are now wrong, for example the removed `AgentExecutor`, the sunset `langchain-community` package, and parameters newer models reject.
- **Provider-agnostic by design.** All model access goes through one small `get_model()` helper, so the whole project can switch between Anthropic, OpenAI and Google by changing one argument.
- **Frameworks earned, not assumed.** You write the tool-calling loop by hand before `create_agent` automates it, and you manage message history yourself before LangChain does it for you.

## Repository layout

```
.
├── README.md
├── article-01-genai-basics-langchain.html      # articles 1 to 15, HTML and Markdown
├── article-01-genai-basics-langchain.md
├── ...
├── article-15-capstone.html
├── article-15-capstone.md
└── tidewave_assistant/                         # the finished capstone project
    ├── .env.example
    ├── requirements.txt
    ├── tidewave_docs/                          # the documentation the assistant searches
    ├── models.py                               # Article 5: get_model()
    ├── documents.py                            # Article 9: document loading
    ├── knowledge_base.py                       # Articles 9 to 11: vector store and retriever
    ├── tools.py                                # Articles 11 and 12: search_docs and account lookup
    ├── triage.py                               # Article 6: structured-output triage
    ├── assistant.py                            # Articles 13 and 14: the agent
    ├── main.py                                 # entry point
    └── tests/                                  # pytest suite with no-key stand-ins
```

## License

MIT.
