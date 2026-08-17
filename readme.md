

```markdown
# AI Answer Evaluator (Rubric & NLI Based)

An automated, non-LLM pipeline for evaluating student answers against multi-question rubrics using **DeBERTa Natural Language Inference (NLI) Cross-Encoders** and deterministic rule matchers. 

The pipeline dynamically fetches model answers, rubrics, and student submissions directly from the target GitHub repository (`ai-question-mapping`), evaluates the answers sentence-by-sentence, and outputs structured JSON reports containing criterion-level score breakdowns.

---

## 🏗️ Project Architecture

```text
evaluator_project/
│
├── output/                         # Output directory generated at runtime
│   └── student1_evaluated.json
│
├── src/
│   ├── __init__.py
│   ├── schemas.py                 # Pydantic data models for input/output JSONs
│   ├── utils.py                   # GitHub fetching (urllib) & file I/O helpers
│   ├── criteria_evaluators.py     # NLI semantic and count-based evaluation handlers
│   └── engine.py                  # DeBERTa Cross-Encoder scoring engine
│
├── requirements.txt               # Dependencies
├── main.py                        # Automated execution entry point
└── README.md

```

---

## ⚡ Key Features

* **Zero LLM Dependency:** Runs completely deterministically using transformer cross-encoders without external API costs or stochastic hallucination risks.
* **100% Automated GitHub Integration:** Downloads `rubric.json`, `model_answers.json`, and all student files in `input/` over HTTPS directly from GitHub using `urllib`.
* **Multi-Criteria Support:**
* `type: "semantic"`: Evaluates conceptual alignment via NLI Entailment vs. Contradiction.
* `type: "minimum_count"`: Checks presence and count of target options/keywords in student answers.


* **Sentence-Level Matching:** Prevents dilution by splitting student responses into individual sentences and extracting peak entailment scores per criterion.

---

## 🚀 Quick Start

### 1. Create and Activate Virtual Environment (`.venv`)

Set up an isolated virtual environment before installing dependencies:

#### Windows (PowerShell)

```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

```

#### Windows (Command Prompt)

```cmd
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\activate.bat

```

#### macOS / Linux

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

```

---

### 2. Installation

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt

```

---

### 3. Execution

Run the automated pipeline:

```bash
python main.py

```

The pipeline will:

1. Connect to GitHub and fetch `rubric.json` and `model_answers.json`.
2. Discover all student files in the GitHub `input/` folder using the GitHub REST API.
3. Process each submission and output evaluation reports inside `output/<student_id>_evaluated.json`.

---

## 📊 Evaluation Logic & Metrics

Semantic criteria use **Natural Language Inference (NLI)** probabilities:

$$\text{Output} = \Big[ P(\text{Contradiction}),\, P(\text{Entailment}),\, P(\text{Neutral}) \Big]$$

* **$P(\text{Entailment}) \ge 0.85$**: Fully Satisfied $\rightarrow$ **Full Marks**
* **$P(\text{Entailment}) \ge \text{Threshold}$**: Partially Satisfied $\rightarrow$ **Scaled Partial Marks**
* **$P(\text{Contradiction}) > 0.60$**: Factually Incorrect $\rightarrow$ **0 Marks**
* **$P(\text{Neutral}) \approx 1.0$**: Missing Concept $\rightarrow$ **0 Marks**

---

## 🔮 Future Model Upgrade Roadmap

To improve evaluation accuracy, reduce latency, or adapt to larger documents, the pipeline can be upgraded with alternative non-LLM models:

### 1. Specialized Reranking Cross-Encoders

* **`BAAI/bge-reranker-large`**: Outstanding performance on dense semantic similarity and key-concept alignment.
* **`BAAI/bge-reranker-v2-m3`**: Multilingual support with fast execution for domain-specific terminology.
* **`cross-encoder/ms-marco-Electra-large`**: Optimized for fine-grained factual accuracy checks.

### 2. Multi-Dataset Fine-Tuned NLI Models

* **`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`**: Trained across multi-NLI datasets (FEVER, ANLI, WanLI). Significantly reduces false neutral readings on complex academic sentences.
* **`facebook/bart-large-mnli`**: Highly stable zero-shot classifier for structural and descriptive criteria.

### 3. Two-Stage Hybrid Pipeline (Embedding + Cross-Encoder)

For multi-paragraph student essays, a two-stage pipeline can be implemented:

1. **Stage 1 (Dense Retrieval):** Filter student responses to top-$K$ relevant sentences using `BAAI/bge-large-en-v1.5` or `all-mpnet-base-v2`.
2. **Stage 2 (NLI Verification):** Evaluate only those $K$ candidate sentences through DeBERTa, eliminating background noise and speeding up execution.

### 4. Custom Fine-Tuned DeBERTa

Fine-tuning `DeBERTa-v3` directly on 100–200 human-graded student answer pairs using PyTorch / Hugging Face `SentenceTransformers` to maximize Quadratic Weighted Kappa (QWK) scores for subject-specific rubrics.

### Current Problems:

Low accuracy of DeBERT NLI model.
