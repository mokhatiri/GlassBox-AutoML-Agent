# GlassBox AutoML

GlassBox AutoML is a transparent machine learning library built from scratch with NumPy. The aim is to provide an end-to-end AutoML pipeline that remains readable, explainable, and easy to debug.

> This is my fork of a 5-person UM6P team project. Upstream: [Gyro0/GlassBox-AutoML-Agent](https://github.com/Gyro0/GlassBox-AutoML-Agent) · Package: [glassbox-automl-agent on PyPI](https://pypi.org/project/glassbox-automl-agent/)

## My contribution (Mohammed Khatiri)

I was the top committer (11 of 33 commits). I built:

- the EDA module: profiler, automatic column typing, correlation and outlier detection
- the matrix helpers, the classification and regression metrics, and the consolidated test suite
- the MCP server that exposes AutoFit to LLM agents, and the IronClaw web demo
- the scikit-learn comparison benchmark below

## Results against scikit-learn

Run on 29 Oct 2026 with `benchmarks/sklearn_comparison.py` and its default train/test split. Every GlassBox model lands within 1% of its scikit-learn equivalent.

**Regression, California housing** (20,640 rows, R²)

| Model | GlassBox | scikit-learn |
| --- | --- | --- |
| Linear regression | 0.640 | 0.640 |
| Decision tree | 0.621 | 0.621 |
| Random forest | 0.691 | 0.686 |
| k-nearest neighbours | 0.709 | 0.709 |

**Classification, stroke prediction** (5,110 rows, accuracy)

| Model | GlassBox | scikit-learn |
| --- | --- | --- |
| Logistic regression | 0.951 | 0.951 |
| Decision tree | 0.948 | 0.949 |
| Random forest | 0.951 | 0.951 |
| Gaussian naive Bayes | 0.571 | 0.571 |
| k-nearest neighbours | 0.947 | 0.947 |

Only 4.9% of the stroke rows are positive, so predicting "no stroke" every time already scores 0.951. This table shows that GlassBox reproduces scikit-learn, not that the models predict strokes well.

## Project Goal

The project covers the full machine learning workflow inside the `glassbox/` package:

- exploratory data analysis
- preprocessing
- models
- evaluation
- hyperparameter optimization
- agent-level AutoFit integration

Core library modules are built from scratch with NumPy only. No Scikit-Learn code belongs inside `glassbox/`.

## Installation

### From PyPI

Install the published package:

```bash
python3 -m pip install glassbox-automl-agent
```

PyPI project page: [glassbox-automl-agent](https://pypi.org/project/glassbox-automl-agent/).

### Manual Installation From Repository

Clone the repository, create and activate a virtual environment, then install dependencies:

```bash
git clone https://github.com/Gyro0/GlassBox-AutoML-Agent.git
cd GlassBox-AutoML-Agent
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you need local development tools:

```bash
python3 -m pip install pytest jupyter
```

## Quick Start

Run the full AutoFit pipeline on the included sample CSV:

```python
from glassbox.agent import auto_fit

report = auto_fit(
    "data/sample.csv",
    target_column="purchased",
    task="auto",
    search="random",
    time_budget=20,
)

print(report["best_model"])
print(report["cv_score"])
print(report["eda_summary"]["overview"])
```

The returned report is JSON-safe and includes:

- EDA overview, numerical profile table, correlations, and outlier rows
- selected task type
- candidate model leaderboard
- best model, best parameters, and cross-validation score
- feature importances or coefficient-style importances when available

## Manual Workflow Example

```python
import numpy as np

from glassbox.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from glassbox.models import RandomForestClassifier
from glassbox.evaluation.classification import classification_report

X_num = np.array([
    [22.0, 32000.0],
    [24.0, np.nan],
    [42.0, 76000.0],
])
X_cat = np.array([["basic"], ["basic"], ["plus"]], dtype=object)
y = np.array([0, 0, 1])

X_num = SimpleImputer(strategy="mean").fit_transform(X_num)
X_num = StandardScaler().fit_transform(X_num)
X_cat = OneHotEncoder().fit_transform(X_cat)
X = np.hstack([X_num, X_cat])

model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
model.fit(X, y)
predictions = model.predict(X)

print(classification_report(y, predictions))
```

## Model Zoo

Classification:

- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GaussianNaiveBayes`
- `KNearestNeighbors(task="classification")`

Regression:

- `LinearRegression`
- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `KNearestNeighbors(task="regression")`

## Demo And Benchmarks

Launch the notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

Run the Scikit-Learn comparison benchmark for regression:

```bash
python benchmarks/sklearn_comparison.py --task regression --csv data/_uploaded.csv --target Delay
```

Run the Scikit-Learn comparison benchmark for classification:

```bash
python benchmarks/sklearn_comparison.py --task classification --csv data/classification.csv --target stroke
```

If `--target` is omitted, the script uses the last column in the CSV as the prediction target.

Scikit-Learn is used only in the benchmark script. The `glassbox/` package itself remains NumPy-only.

## Repository Structure

```text
GlassBox-AutoML-Agent/
|-- .github/
|   `-- pull_request_template.md
|-- glassbox/
|   |-- agent/
|   |-- eda/
|   |-- evaluation/
|   |-- models/
|   |-- optimization/
|   |-- preprocessing/
|   `-- utils/
|-- tests/
|-- notebooks/
|-- benchmarks/
|-- data/
|-- README.md
|-- pyproject.toml
`-- requirements.txt
```

## Testing

Run the test suite from the repository root:

```bash
python3 -m pytest -q
```

## IronClaw / MCP Tool

The library exposes a single tool, `auto_fit`, through three surfaces:

- **`glassbox.agent.mcp_server`** — a FastMCP server over stdio, the IronClaw deployment target.
- **`glassbox.agent.mcp_tool`** — a JSON-in/JSON-out CLI shim for scripted/sandbox testing.
- **`mcp.json`** — reference manifest documenting the tool schema (not consumed by IronClaw directly).

### Register with IronClaw

After SSHing to your IronClaw box and `pip install -e .[mcp]`:

```bash
ironclaw mcp add glassbox \
  --transport stdio \
  --command python \
  --arg -m --arg glassbox.agent.mcp_server
```

IronClaw stores the registration in `~/.ironclaw/mcp-servers.json` and spawns the server over stdio whenever the agent calls the tool. Verify with `ironclaw mcp list`.

### Run the tool directly (path mode)

```bash
python -m glassbox.agent.mcp_tool --input '{
  "csv_path": "data/sample.csv",
  "target_column": "purchased",
  "task": "auto",
  "search": "random",
  "time_budget": 20
}'
```

Run the tool inside a sandbox where there is no host filesystem (bytes mode):

```bash
python -c "import base64,json,sys; \
  print(json.dumps({'csv_b64': base64.b64encode(open('data/sample.csv','rb').read()).decode(), \
                    'target_column':'purchased'}))" \
  | python -m glassbox.agent.mcp_tool
```

The response is a single JSON object: `{"ok": true, "report": {...}}` on success, or `{"ok": false, "error": "..."}` on failure. The report includes an `explanation` array of short bullets the agent can repeat back to the user.

### Agent private key

Never commit the IronClaw agent private key. Use one of:

1. Environment variable: `export IRONCLAW_AGENT_PRIVATE_KEY=...` (or place it in a local `.env` — already gitignored).
2. The IronClaw CLI's own keystore at `~/.ironclaw/credentials` (preferred for production).

The `glassbox/` package itself never reads the key; only the IronClaw runtime does, when registering the agent.
