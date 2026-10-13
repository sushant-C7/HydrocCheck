# HydroCheck AI — Agentic AI Upgrade

This version keeps the existing stacking classifier and SHAP code, and adds:

1. **Isolation Forest anomaly detection** (runs locally; no LLM/API key required).
2. **A tool-calling AI agent** that chooses among local prediction, anomaly detection, range checks, and reference retrieval.
3. **A standalone RAG research assistant** that answers reference questions without requiring sample measurements.
4. **Visible sources and execution traces** showing retrieved documents and tools actually called.
5. **Input validation** for JSON requests, non-numeric values, infinities, and negative measurements.

The agent is a single orchestrator, not a collection of unnecessary sub-agents. The model-based prediction and anomaly detector run locally. The language-model-based agent requires an OpenAI API key and internet access. API use is billed separately from ChatGPT subscriptions; check your API account's billing and limits before use.

## Project structure changes

- `app/app.py`: refactored prediction into a reusable function, added `/agent`, and included anomaly results in `/predict`.
- `app/templates/index.html`: added the AI Agent panel, tool-execution trace, and anomaly result.
- `src/anomaly.py`: Isolation Forest training, loading, anomaly scoring, and a training-range checker.
- `src/train_anomaly.py`: optional explicit command to train the anomaly detector.
- `src/agent.py`: bounded LLM tool-calling loop.
- `.env.example`: sample configuration. Never publish your real `.env` file or API key.

## Windows PowerShell setup

Run these commands from the `hydrocheck` project directory, the one that contains `app`, `src`, `models`, and `data`.

### 1. Activate the virtual environment

If the old project's virtual environment is still present:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead run commands with `python -m pip ...` and `python -m ...` from the same project directory, or use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for this terminal session only.

### 2. Install the two new dependencies

If your existing virtual environment already runs the original HydroCheck project, keep it active and install only the additions:

```powershell
python -m pip install openai python-dotenv
```

This avoids unnecessarily upgrading the scientific Python packages used to serialize your existing `models/stack.pkl` file.

If you are setting the whole project up in a fresh environment instead, install the full requirements file:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a fresh environment, use a Python version supported by all the listed ML packages. Keep scikit-learn and XGBoost compatible with the versions used to train the bundled `stack.pkl` model.

### 3. Configure the LLM key (only needed for the agent)

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace `put_your_api_key_here` with your own OpenAI API key and save. Keep `.env` private and do not upload it to GitHub or include it in a submission archive. The app still starts without a key, and ordinary prediction/anomaly checks still run; the Agent panel will report that it is not configured.

### 4. Train the anomaly detector (optional)

The application trains it automatically on first launch if `models/isolation_forest.pkl` is absent. You can train it manually instead:

```powershell
python -m src.train_anomaly
```

### 5. Start the app

From the project root (not from inside `app`):

```powershell
python -m app.app
```

Open `http://127.0.0.1:5000` in your browser.

### 6. Use it

1. Enter the nine measurement values.
2. Click **Analyze Water Sample** to run the existing classifier, SHAP component explanation, and the local anomaly detector.
3. In **HydroCheck AI Agent**, enter a task such as:
   - `Analyze this sample and explain the model prediction.`
   - `Check for anomalies and values outside the training-data range.`
   - `Give me a complete analysis and explain the limitations.`
4. Click **Ask HydroCheck AI Agent**. The webpage shows the final answer and the actual tool calls made by the LLM.

## Test the API

With the server running, in a second PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Example prediction request:

```powershell
$body = @{
  ph = 7.1
  Hardness = 190
  Solids = 20000
  Chloramines = 7
  Sulfate = 330
  Conductivity = 420
  Organic_carbon = 14
  Trihalomethanes = 66
  Turbidity = 3.9
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType 'application/json' -Body $body
```

Example agent request:

```powershell
$agentBody = @{
  message = 'Analyze this sample, explain the prediction, and check for unusual values.'
  measurements = @{
    ph = 7.1
    Hardness = 190
    Solids = 20000
    Chloramines = 7
    Sulfate = 330
    Conductivity = 420
    Organic_carbon = 14
    Trihalomethanes = 66
    Turbidity = 3.9
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/agent -ContentType 'application/json' -Body $agentBody
```

## Important interpretation notes

- **Anomaly does not mean unsafe.** Isolation Forest flags observations that are unusual relative to the training data.
- **Training range is not a regulatory limit.** The range checker only says whether a measurement falls outside the values represented in the available dataset.
- **The probability is a model output, not proof of safety.** The model needs formal calibration and independent validation before any high-stakes use.
- **SHAP scope:** this project currently explains the XGBoost component of the stacking ensemble, not the full ensemble's combined decision. The UI calls this out explicitly.
- **Do not claim the agent improves classification accuracy** unless you evaluate that claim. The agent orchestrates existing tools and explains their outputs; it does not retrain the classifier.

## Common issues

- **Agent says API key is not configured:** confirm `.env` exists in the project root, the key is valid, and restart the Flask process.
- **401/403 API error:** check that the API key belongs to a project with model access.
- **429 / quota error:** check API usage limits and billing.
- **Could not import `src` or `app`:** make sure the terminal's current directory is the project root and start with `python -m app.app`.
- **Model pickle compatibility errors:** use a Python/scikit-learn environment compatible with the environment in which the existing `models/stack.pkl` was trained. Do not retrain the ensemble unless you intend to regenerate the model files and evaluate it again.

## RAG water-quality research assistant (added)

The webpage now includes a standalone **RAG Water Quality Research Assistant** panel, and the Agent panel can also invoke the same **retrieval-augmented generation (RAG)** tool as part of a sample investigation. It searches the local `knowledge_base/` collection using scikit-learn TF-IDF (terms and bigrams), sends the most relevant passages and source metadata to the LLM, and shows retrieved sources in the webpage. It does not browse the live web at query time and does not need an additional vector-database or embedding-service account.

The included notes point to:

- WHO's *Guidelines for Drinking-water Quality*, fourth edition incorporating the first, second and third addenda (official publication page: https://www.who.int/publications/i/item/9789240121225; published 17 June 2026).
- The BIS official catalogue/search for IS 10500, *Drinking Water — Specification*: https://lims.bis.gov.in/home/search_is_number/?is_number__doc_no=10500.
- The US EPA's official National Primary and Secondary Drinking Water Regulations overview: https://www.epa.gov/ground-water-and-drinking-water/national-primary-drinking-water-regulations.
- Project-specific notes describing HydroCheck's model limitations and terms used in the data.

These local files are curated summaries, not full copies of the standards or publications. They intentionally avoid reproducing numerical limit tables. For exact limits and compliance, check the current official source and the right jurisdiction. The agent is instructed not to treat US EPA requirements as Indian law, or to treat a model score as a compliance certificate.

### Try RAG

In the RAG Research Assistant panel, try one of these:

- `Explain the WHO drinking-water risk-management framework and cite your sources.`
- `What is BIS IS 10500, and can HydroCheck certify that this sample complies with it?`
- `Are US EPA drinking-water rules applicable in India? Explain with sources.`
- `Explain chloramines and trihalomethanes using the local reference library.`

The RAG panel trace should show `search_water_quality_knowledge`. Its sources section lists the documents returned by the retriever. The response is synthesized by the configured LLM, so the standalone research panel still needs an OpenAI API key. The local `/knowledge/search` endpoint works without an API key. The sample-analysis Agent can use this same knowledge-search tool alongside its ML tools.

### Test RAG without an LLM API call

The local retrieval endpoint runs without an OpenAI API key or supplied measurements:

```powershell
$body = @{ query = 'What is BIS IS 10500 in India?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/knowledge/search -ContentType 'application/json' -Body $body
```

Health status includes how many local documents and passages were indexed:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

The standalone RAG question-answering UI uses the `/research` endpoint. For a direct JSON request, use:

```powershell
$body = @{ question = 'Are US EPA drinking-water rules applicable in India?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/research -ContentType 'application/json' -Body $body
```

### Add or update knowledge

Add `.md` or `.txt` documents to `knowledge_base/` using the metadata template in `knowledge_base/README.md`, then restart Flask. Use official, current sources and identify the jurisdiction. The retriever is local keyword/bigram similarity rather than semantic embeddings; for better retrieval at larger scale, consider evaluating an embedding model and a vector index in a later iteration.
