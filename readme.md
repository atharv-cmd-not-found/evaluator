
# AI Question & Rubric Evaluation Pipeline

An automated, multi-model academic answer evaluation and benchmarking platform. This system grades student answers against structured question rubrics and reference model answers using a hybrid framework combining **Cross-Encoder Neural Networks** (Natural Language Inference, Semantic Textual Similarity, and Passage Rerankers) with **Local Open-Source Large Language Models (LLMs)** served via Ollama.



## Key Features

- **Multi-Model Benchmark Architecture**: Seamlessly evaluates and compares grading performance across DeBERTa (NLI), RoBERTa (STS), BGE Reranker, and local instruction-tuned LLMs (Qwen 2.5, Mistral, Phi-3.5).
- **Fine-Grained Criterion Diagnostics**: Evaluates responses on a criterion-by-criterion basis using token entailment scores, contradiction penalties, semantic similarity thresholds, and minimum entity match counts.
- **Robust Local Execution**: Runs inference entirely on local hardware without sending proprietary data to external cloud APIs.
- **Dynamic CLI & Interactive Fallback**: Supports command-line flag execution for automated pipelines, alongside an interactive terminal interface for ad-hoc grading.
- **Dual-Layer Reporting Engine**: Produces individual per-student evaluation JSON schemas and combined comparative CSV benchmark reports for aggregate analysis.
- **Error-Tolerant Engine**: Auto-detects offline LLM services and gracefully falls back to available neural Cross-Encoder models without halting evaluation pipelines.



## Tech Stack & Dependencies

### Core Technologies
- **Python 3.10+**: Core programming environment.
- **PyTorch (`torch`)**: Deep learning computation and tensor operations.
- **Hugging Face Transformers & Sentence-Transformers**: Cross-Encoder sequence classification and embedding models.
- **Pydantic V2**: Strict schema enforcement, input validation, and structured serialization.
- **Ollama Engine**: Local execution runtime for quantized Large Language Models.
- **Pandas & NumPy**: Analytical data structures, vectorized scoring, and tabular data export.
- **Requests**: HTTP networking client for communicating with Ollama REST API endpoints.

### Complete Dependency Manifest (`requirements.txt`)
```text
torch>=2.2.0
sentence-transformers>=3.0.0
transformers>=4.40.0
pydantic>=2.5.0
pandas>=2.1.0
numpy>=1.24.0
requests>=2.31.0
tabulate>=0.9.0

```

---

## Local LLM Setup (Ollama)

The pipeline integrates with open-source Large Language Models using [Ollama](https://ollama.com/) over its local HTTP REST API (`http://localhost:11434/api/generate`).

### 1. Start the Ollama Service

Ensure the Ollama background daemon is active:

```bash
# Start the Ollama server process
ollama serve

```

*Note for Windows Users: If installed via the desktop installer, Ollama runs automatically in the background system tray. Verify server readiness by querying the endpoint:*

```bash
curl http://localhost:11434
# Expected output: "Ollama is running"

```

### 2. Pull Required Models

Download the supported evaluation models locally:

```bash
# Primary recommended model (7B parameters, ~4.7 GB)
ollama pull qwen2.5:7b

# Lightweight model optimized for CPU execution (~1.9 GB)
ollama pull qwen2.5:3b

# Secondary benchmark models
ollama pull mistral:7b
ollama pull phi3.5:latest

```

### 3. Verify Model Installation

```bash
# View list of installed models
ollama list

# Run a quick sanity check
ollama run qwen2.5:7b "Evaluate: What is 2+2? Reply in JSON."

```

---

## Installation & Setup

```bash
# 1. Clone the project repository
git clone https://github.com/atharv-cmd-not-found/evaluator.git
cd evaluator

# 2. Create and activate a Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

```

---

## Running the Evaluator

The evaluation pipeline can be launched in two modes: **Flag-Based CLI Execution** or **Interactive Terminal Execution**.

### 1. Flag-Based Execution (Recommended)

Pass custom paths for the rubric, reference answers, student submissions, and output directories using explicit command-line flags:

```bash
# Evaluate a single student submission
python main.py -r input/rubric.json -m input/model_answers.json -s input/student1.json

# Evaluate multiple student submissions simultaneously
python main.py -r input/rubric.json -m input/model_answers.json -s input/student1.json input/student2.json input/student3.json

# Export to a custom output directory
python main.py -r custom_rubric.json -m custom_answers.json -s input/student1.json -o custom_output_dir

```

#### CLI Flag Reference

| Flag | Long Flag | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `-r` | `--rubric` | `str` | `input/rubric.json` | Path to the structured rubric definition JSON file. |
| `-m` | `--model_answers` | `str` | `input/model_answers.json` | Path to the reference/model answers JSON file. |
| `-s` | `--students` | `list` | `None` | One or more space-separated paths to student JSON files. |
| `-o` | `--output_dir` | `str` | `output` | Target directory where JSON and CSV reports are written. |

---

### 2. Interactive Terminal Execution

Run the script without passing flags to trigger the interactive prompt:

```bash
python main.py

```

* When prompted:
```text
Enter path(s) to student JSON file(s) [or leave blank to scan 'input/']:

```


* **Press Enter** to automatically scan and evaluate all valid student `.json` files inside the `input/` folder.
* Or manually type a comma-separated or space-separated list of relative file paths (e.g., `input/student1.json, input/student2.json`).



---

## Project Structure

```text
├── input/
│   ├── rubric.json                # Grading criteria, weightages, and thresholds
│   ├── model_answers.json         # Reference answers for evaluation context
│   └── student1.json              # Student submission JSON file
├── output/
│   ├── comparative_students_summary.csv    # Consolidated scorecards across all models
│   ├── comparative_criteria_benchmark.csv  # Granular criteria metrics and diagnostics
│   └── student1_*_evaluated.json           # Individual detailed student JSON outputs
├── src/
│   ├── criteria_evaluators.py     # NLI, STS, and minimum count grading logic
│   ├── engine.py                  # Evaluation orchestrator across all model types
│   ├── llm_evaluator.py           # Ollama API client and structured prompting handler
│   ├── schemas.py                 # Pydantic data contracts for evaluation results
│   └── utils.py                   # File I/O, sentence splitting, and path resolution
├── main.py                        # Program entry point with CLI and interactive handling
├── requirements.txt               # Locked project dependencies
└── README.md                      # Project documentation

```

```

```