"""Tool-calling HydroCheck agent with local RAG retrieval.

The LLM chooses from bounded, read-only tools. The application's ML models and
knowledge retrieval run locally; the LLM synthesizes tool outputs into a report.
"""

import json
import os


class AgentConfigurationError(RuntimeError):
    """Raised when the agent cannot be configured."""


def _function_tool(name, description):
    """Define a no-argument local analysis tool for OpenAI tool calling."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def _knowledge_search_tool():
    """Define a query-argument tool for searching the local reference collection."""
    return {
        "type": "function",
        "function": {
            "name": "search_water_quality_knowledge",
            "description": (
                "Search the local, curated reference collection for information about drinking-water "
                "guidance, Indian BIS standards, US EPA regulations, measurement terms, and HydroCheck "
                "model limitations. Use for questions about guidelines, regulations, health context, "
                "definitions, or when the user asks for evidence or sources. The returned passages include "
                "jurisdiction labels and official source URLs. This tool does not search the live web."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def run_hydrocheck_agent(
    message,
    measurements,
    prediction_fn,
    anomaly_fn,
    range_fn,
    knowledge_fn=None,
):
    """Run a bounded tool-calling loop and return answer, source list and real tool trace."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "put_your_api_key_here":
        raise AgentConfigurationError(
            "OPENAI_API_KEY is not configured. Add your API key to the .env file, then restart the server."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AgentConfigurationError(
            "The OpenAI Python package is not installed. Run: python -m pip install openai"
        ) from exc
    client = OpenAI(api_key=api_key)

    tools = [
        _function_tool(
            "run_water_prediction",
            "Run the existing local HydroCheck stacking ML model and return its prediction, threshold, and SHAP feature contributions for the current sample. Use this for potability assessment or when explaining a model result.",
        ),
        _function_tool(
            "detect_input_anomaly",
            "Run the local Isolation Forest model to check whether the current sample looks unusual compared with the training dataset. Use when the user asks about anomalies or for a comprehensive analysis.",
        ),
        _function_tool(
            "check_training_ranges",
            "Check if current measurement values are outside the minimum/maximum ranges represented in the training dataset. This checks model familiarity, not regulatory compliance. Use for comprehensive analyses or unfamiliar values.",
        ),
        _knowledge_search_tool(),
    ]

    handlers = {
        "run_water_prediction": lambda: prediction_fn(measurements),
        "detect_input_anomaly": lambda: anomaly_fn(measurements),
        "check_training_ranges": lambda: range_fn(measurements),
    }

    system_message = """You are HydroCheck AI Agent, a careful technical assistant for a water-quality ML prototype.
You can call local model tools and search a curated local knowledge base. Select tools according to the user's task. For a potability assessment or explanation of this sample's model output, call run_water_prediction. If the user asks whether readings are unusual or requests a full analysis, call detect_input_anomaly. For a full analysis or concerns about unfamiliar values, call check_training_ranges. When the user asks about water-quality meaning, guidelines, regulations, health context, evidence, or sources, call search_water_quality_knowledge. For a request to compare this sample to guidance, use the prediction tool if relevant and retrieve reference passages; do not claim that the prediction is a compliance determination.

Grounding rules: For claims based on retrieved documents, use only the returned passages and cite them using Markdown links with the exact returned source URLs, e.g. [Document title](https://...). Include the jurisdiction of standards and guidance. WHO is international guidance; BIS is relevant to India; US EPA pages describe United States rules and must not be presented as Indian law. If no relevant passage is returned, say the local knowledge base did not find a reliable answer and direct the user to the official source; do not invent limits, dates, health effects, or citations. The included knowledge notes are summaries, not the complete legal or technical publications. For exact numerical limits and regulatory compliance, tell the user to verify the current official standard and applicable parameter list rather than inferring an unprovided value.

Never invent model outputs, tool results, regulatory limits, or laboratory measurements. Explain that an Isolation Forest anomaly or an out-of-training-range value is not proof of unsafe water. Do not say water is guaranteed safe to drink; this ML prototype cannot replace certified laboratory testing. Clearly distinguish model predictions from retrieved guidance and from legal requirements. Give a concise report with findings, citations when retrieved, limitations, and sensible verification next steps. The tools are read-only and only analyse the supplied sample or local reference collection."""

    measurement_json = json.dumps(
        {key: measurements.get(key) for key in measurements},
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": (
                f"User request: {str(message).strip()[:1200]}\n\n"
                f"Current water sample measurements (JSON): {measurement_json}\n"
                "Use available tools when needed; base all sample-specific claims on actual tool results. "
                "When the request needs guidance or evidence, retrieve relevant passages before answering."
            ),
        },
    ]

    trace = []
    all_sources = []
    seen_sources = set()
    max_rounds = 6

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1000,
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            answer = (assistant_message.content or "The agent completed without a text response.").strip()
            return {
                "reply": answer,
                "tool_trace": trace,
                "sources": all_sources,
                "model": model_name,
                "rag_enabled": True,
            }

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            try:
                if name == "search_water_quality_knowledge":
                    if knowledge_fn is None:
                        raise ValueError("Knowledge retrieval is not configured in this app instance.")
                    arguments = json.loads(tool_call.function.arguments or "{}")
                    query = str(arguments.get("query", "")).strip()[:600]
                    if not query:
                        raise ValueError("The knowledge search query was empty.")
                    result = knowledge_fn(query, top_k=4)
                    for source in result.get("sources", []):
                        source_key = (source.get("title", ""), source.get("source_url", ""))
                        if source_key not in seen_sources:
                            all_sources.append(source)
                            seen_sources.add(source_key)
                    trace.append({
                        "tool": name,
                        "status": "completed" if result.get("found") else "no_results",
                        "summary": _summarize_tool_result(name, result),
                    })
                else:
                    if name not in handlers:
                        raise ValueError(f"Tool is not permitted: {name}")
                    # Analysis functions operate only on validated measurements from this request.
                    result = handlers[name]()
                    trace.append({
                        "tool": name,
                        "status": "completed",
                        "summary": _summarize_tool_result(name, result),
                    })
                tool_content = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as exc:
                trace.append({"tool": name, "status": "failed", "summary": str(exc)[:240]})
                tool_content = json.dumps({"error": str(exc)[:400]}, ensure_ascii=False)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content,
            })

    # Ask the model for a final synthesis after the bounded tool loop.
    messages.append({
        "role": "user",
        "content": "Provide the final concise report using only tool results already supplied. Cite retrieved sources using their exact URLs. Do not request more tools.",
    })
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tool_choice="none",
        temperature=0.2,
        max_tokens=1000,
    )
    final_text = response.choices[0].message.content or "The agent reached its tool-call limit before completing a report."
    return {
        "reply": final_text.strip(),
        "tool_trace": trace,
        "sources": all_sources,
        "model": model_name,
        "rag_enabled": True,
    }


def _summarize_tool_result(name, result):
    """Create a concise execution log without exposing full tool payloads in the UI."""
    if isinstance(result, dict) and result.get("error"):
        return str(result["error"])[:200]

    if name == "run_water_prediction":
        if result.get("source") == "rule_gate":
            return "Rule-based safety checks flagged the sample."
        label = "potable" if result.get("potable") else "not potable"
        return f"Local stacking model returned {label}; probability={result.get('probability')}."

    if name == "detect_input_anomaly":
        return result.get("label", "Isolation Forest analysis completed.")

    if name == "check_training_ranges":
        count = len(result.get("details", []))
        return f"{count} measurement(s) outside training-data ranges."

    if name == "search_water_quality_knowledge":
        if not result.get("found"):
            return result.get("message", "No relevant passages found in the local knowledge base.")
        count = len(result.get("passages", []))
        sources = len(result.get("sources", []))
        return f"Retrieved {count} relevant passage(s) from {sources} source document(s) using local TF-IDF search."

    return "Tool completed."


def run_rag_research_assistant(question, knowledge_fn):
    """Answer a reference question using local RAG retrieval, without sample inputs."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "put_your_api_key_here":
        raise AgentConfigurationError(
            "OPENAI_API_KEY is not configured. Add your API key to the .env file, then restart the server."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AgentConfigurationError(
            "The OpenAI Python package is not installed. Run: python -m pip install openai"
        ) from exc
    client = OpenAI(api_key=api_key)

    system_message = """You are HydroCheck's RAG Water Quality Research Assistant. Answer the user's water-quality research question using passages retrieved from the local knowledge base. Search before answering substantive questions. The documents are curated summaries of official or project-specific sources, not complete publications, and the retriever does not browse the live web.

Cite external source-based claims with Markdown links using only exact URLs returned by the retrieval tool. Name the jurisdiction. WHO is international guidance, BIS is relevant to India, and US EPA pages describe US rules only; never present US rules as Indian law. If retrieved sources do not support a claim, say so instead of guessing. Do not invent numeric guideline values, legal requirements, citations or health effects. When asked for exact limits or compliance, explain that the current complete official standard and appropriate laboratory results must be checked. HydroCheck model predictions, anomaly scores and training-data ranges are not regulatory compliance determinations. Use clear language and state uncertainty or missing information."""

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": (
                f"Research question: {str(question).strip()[:1200]}\n\n"
                "Retrieve relevant passages and answer with source citations. If the corpus has no relevant passage, say that clearly."
            ),
        },
    ]

    trace = []
    all_sources = []
    seen_sources = set()
    max_rounds = 4

    for round_index in range(max_rounds):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=[_knowledge_search_tool()],
            tool_choice="required" if round_index == 0 else "auto",
            temperature=0.2,
            max_tokens=1000,
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            return {
                "reply": (assistant_message.content or "The research assistant completed without a text response.").strip(),
                "tool_trace": trace,
                "sources": all_sources,
                "model": model_name,
                "retrieval_method": "TF-IDF term and bigram similarity",
            }

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            try:
                if name != "search_water_quality_knowledge":
                    raise ValueError(f"Tool is not permitted in research mode: {name}")
                arguments = json.loads(tool_call.function.arguments or "{}")
                search_query = str(arguments.get("query", "")).strip()[:600]
                if not search_query:
                    raise ValueError("The knowledge search query was empty.")
                result = knowledge_fn(search_query, top_k=4)
                for source in result.get("sources", []):
                    key = (source.get("title", ""), source.get("source_url", ""))
                    if key not in seen_sources:
                        all_sources.append(source)
                        seen_sources.add(key)
                trace.append({
                    "tool": name,
                    "status": "completed" if result.get("found") else "no_results",
                    "summary": _summarize_tool_result(name, result),
                })
                tool_content = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as exc:
                trace.append({"tool": name, "status": "failed", "summary": str(exc)[:240]})
                tool_content = json.dumps({"error": str(exc)[:400]}, ensure_ascii=False)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content,
            })

    messages.append({
        "role": "user",
        "content": "Provide a final answer using only retrieved passages so far. Cite URLs returned by the retrieval tool and mention any missing evidence. Do not call more tools.",
    })
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tool_choice="none",
        temperature=0.2,
        max_tokens=1000,
    )
    final_text = response.choices[0].message.content or "The research assistant reached its tool-call limit before answering."
    return {
        "reply": final_text.strip(),
        "tool_trace": trace,
        "sources": all_sources,
        "model": model_name,
        "retrieval_method": "TF-IDF term and bigram similarity",
    }
