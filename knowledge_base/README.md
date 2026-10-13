# HydroCheck RAG knowledge base

The Markdown files in this folder are locally stored, concise, curated notes. At runtime HydroCheck indexes their text with scikit-learn TF-IDF and retrieves the most relevant passages for a query. The standalone RAG Research Assistant synthesizes those passages with the configured LLM, while `/knowledge/search` exposes retrieval-only results. The LLM receives those passages, metadata and source URLs to generate a source-grounded answer.

## Included sources
- WHO: Guidelines for Drinking-water Quality, fourth edition incorporating the first, second and third addenda (published 17 June 2026).
- BIS: official catalogue entry/search for IS 10500:2012, Drinking Water — Specification (Second Revision).
- US EPA: official National Primary and Secondary Drinking Water Regulations overview.
- HydroCheck-specific model limitations and a glossary.

The Markdown files are summaries, not full copies of the source publications. The retrieval code does not browse the web at query time. Keep the source URLs and publication details current, and consult the live official document for exact values or regulatory decisions. Do not add secrets or personal information to this folder.

## Adding a document
Add a `.md` or `.txt` file. For citations, include simple metadata at the top in this format:

```text
---
title: Descriptive document title
publisher: Publishing organization
jurisdiction: Country, region, or global guidance
published: Publication date or version
source_url: https://official.example/document
---

Document content goes here.
```

Restart the Flask app after modifying the knowledge files; the TF-IDF index is loaded into memory at startup.
