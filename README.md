# AgentX — Tool-Using AI Agent with Missing-Data Handling and Duplicate Response Prevention

AgentX is a Python-based AI agent that combines a language model with deterministic tools to answer questions, retrieve information, perform calculations, and interact with web content. It uses LangGraph to manage the agent workflow, Groq for language-model inference, Pydantic for structured validation, and Playwright for browser automation.

The project focuses on three important capabilities: autonomous tool usage, transparent handling of missing information, and detection of identical consecutive responses.

## 1. Internship Milestone Requirements

| Requirement | Implementation | Verification |
|---|---|---|
| AI agent using at least two tools | Tool registry and individual tool implementations | Tool and workflow tests |
| Complete runnable source code | Python application, configuration, tools, and dependencies | Run the application and test suite |
| Missing-information handling | Structured tool results and `NO_RESULTS` status | `tests/test_missing_data.py` |
| Duplicate consecutive response prevention | Session history manager and duplicate validation in the agent workflow | `tests/test_duplicate_responses.py` |
| Demonstration and automated testing | Interactive CLI or terminal interface | Run the application and pytest |

## 2. Tools and Capabilities

AgentX integrates several tools that allow it to perform tasks beyond generating answers from the language model alone.

- **Calculator:** Evaluates supported mathematical expressions using a restricted Python AST-based evaluator.
- **Knowledge Base Search:** Searches stored project documentation and returns matching records.
- **Web Search:** Retrieves search results and associated source information.
- **Browser Automation:** Uses Playwright to inspect web pages and extract relevant content.
- **Current Time:** Retrieves current time information.
- **Model Fallback:** Attempts configured alternative models when supported errors prevent the preferred model from responding.

The language model determines when a tool is needed, while the agent executes the selected tool and incorporates its result into the response.

## 3. Missing-Information Handling and Hallucination Reduction

AgentX uses structured tool results to distinguish successful retrieval from missing information and execution errors.

When a tool cannot find relevant records, it returns a `NO_RESULTS` status rather than presenting invented information as retrieved evidence. The agent is instructed to acknowledge missing evidence and avoid unsupported factual claims. This approach reduces hallucinations for tool-dependent questions, although it cannot guarantee that every generated statement is correct.

**Implementation references:**
- `agentx/graph.py` — tool execution and result handling.
- `agentx/schemas.py` — structured tool-result status definitions.
- `tests/test_missing_data.py` — tests for missing-information behavior.

## 4. Duplicate Consecutive Response Prevention

AgentX includes a session-based mechanism for identifying identical consecutive answers.

The response manager normalizes generated responses and compares them with the previous response stored for the same session. When a duplicate is detected, the workflow can request a rephrased answer while preserving the verified facts. A bounded retry mechanism prevents repeated rephrasing attempts from continuing indefinitely.

**Implementation references:**
- `agentx/response_manager.py` — session response history and duplicate comparison.
- `agentx/graph.py` — duplicate-response validation and workflow routing.
- `tests/test_duplicate_responses.py` — tests for duplicate detection and handling.

The mechanism is intended to prevent unnecessary repetition without encouraging the agent to invent new facts merely to produce different wording.

## 5. Architecture

AgentX uses a graph-based workflow to coordinate model inference, tool execution, and final response validation.

1. Receive and prepare the user's request.
2. Invoke the configured language model.
3. Determine whether a tool call is required.
4. Execute the requested tool and validate its result.
5. Return tool evidence to the model when additional reasoning is needed.
6. Validate the final response for consecutive duplication.
7. Deliver the response and preserve conversation state where configured.

LangGraph manages the workflow transitions and bounded execution loops. Individual Python tools handle their respective operations.

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application implementation |
| Groq SDK | Language-model inference |
| LangGraph | Agent workflow and state transitions |
| Pydantic | Data and result validation |
| Playwright | Browser automation |
| Textual and Rich | Interactive terminal interface |
| pytest | Automated testing |
| SQLite | Conversation checkpoint persistence, where configured |

## 7. Installation and Configuration

### Prerequisites

- Python 3.10 or 3.11.
- Git.
- A Groq API key for live model inference.
- Google Chrome for features that require visible Chrome automation.

### Install dependencies

Clone the repository or open the project directory, then run:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
playwright install chromium
```

Create a local `.env` file based on `.env.example` and configure the required credentials and settings.

**Security:** Never commit `.env` files, API keys, passwords, or other secrets to the repository.

## 8. Running the Agent

### Interactive terminal interface

```powershell
python run_ui.py
```

### Command-line interface

```powershell
python main.py
```

### Single-query execution

```powershell
python main.py "What is 45 * 12 + 160?"
```

Use the entry point supported by your actual project configuration.

## 9. Demonstration Examples

### Example A: Calculator

**Input:**

```text
What is ((45 * 12) + 160) / 4?
```

**Expected mathematical result:**

```text
175
```

During the demonstration, show the tool execution trace and the actual response produced by AgentX.

### Example B: Knowledge Base Retrieval

**Input:**

```text
According to the stored project documentation, what security
measures does AgentX use to prevent SSRF?
```

Show the knowledge-base search result and explain how the agent uses retrieved information to construct its answer. The exact response depends on the records actually present in the knowledge base.

### Example C: Missing Information

**Input:**

```text
What is the specification for the fictional quantum warp drive
in AgentX? If the information is unavailable, say that it was
not found. Do not guess.
```

If the knowledge base has no matching record, the expected behavior is to acknowledge that the information could not be found rather than invent a specification.

### Example D: Duplicate Consecutive Answers

Ask the same question twice in the same session, for example:

```text
What is the capital of France?
```

Observe whether the agent detects an identical consecutive response and invokes its duplicate-handling mechanism. The exact output depends on the implementation and model response.

## 10. Automated Testing

Run the full test suite from the project root:

```powershell
python -m pytest tests/ -v
```

Run the missing-data tests:

```powershell
python -m pytest tests/test_missing_data.py -v
```

Run the duplicate-response tests:

```powershell
python -m pytest tests/test_duplicate_responses.py -v
```

The terminal output provides the actual results. Record the number of passed, failed, and skipped tests from the test run you perform before submitting the project.

## 11. Project Structure

```text
AgentX/
├── agentx/
│   ├── tools/
│   │   ├── calculator.py
│   │   ├── knowledge_base.py
│   │   ├── browser.py
│   │   ├── search.py
│   │   └── time_tool.py
│   ├── safety/
│   │   └── url_validation.py
│   ├── graph.py
│   ├── agent.py
│   ├── fallback.py
│   ├── response_manager.py
│   ├── schemas.py
│   └── tool_registry.py
├── data/
│   └── knowledge_base.json
├── tests/
│   ├── test_missing_data.py
│   ├── test_duplicate_responses.py
│   ├── test_calculator.py
│   └── ...
├── main.py
├── run_ui.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

This is a representative structure based on the supplied project description. Ensure it matches the files in the final repository.

## 12. Limitations

- Live model inference depends on provider availability, configured credentials, and applicable rate limits.
- Web search and browser extraction depend on external websites and network availability.
- Knowledge-base answers depend on the completeness and quality of stored records.
- Duplicate detection identifies matching consecutive responses; it does not guarantee that semantically similar answers will always be detected.
- Model-generated responses can still contain errors, so retrieved evidence and test results should be evaluated carefully.

## 13. Conclusion

AgentX demonstrates a tool-using AI agent architecture with structured tool execution, missing-information handling, and duplicate consecutive response prevention. The project combines Python tools with a graph-based workflow and automated tests to make agent behavior easier to inspect and verify.

The implementation and test results should be reviewed against the internship milestone requirements before final submission.

## Project Metadata

- **Project:** AgentX
- **Category:** AI Agent Engineering / Python
- **Primary language:** Python
- **Testing framework:** pytest
- **License:** MIT, if the repository includes a valid MIT license.