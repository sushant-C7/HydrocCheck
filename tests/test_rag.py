from src.rag import knowledge_base_status, search_knowledge


def test_knowledge_base_loads_documents():
    status = knowledge_base_status()
    assert status["documents_indexed"] >= 4
    assert status["passages_indexed"] >= status["documents_indexed"]


def test_india_query_retrieves_bis_source():
    result = search_knowledge("What is India's BIS IS 10500 drinking water standard?", top_k=4)
    assert result["found"] is True
    assert any("BIS IS 10500" in source["title"] for source in result["sources"])
    assert any(source["jurisdiction"] == "India" for source in result["sources"])


def test_jurisdiction_warning_retrieves_epa_and_india_sources():
    result = search_knowledge("Are US EPA limits legally applicable to drinking water in India?", top_k=5)
    jurisdictions = {source["jurisdiction"] for source in result["sources"]}
    assert result["found"] is True
    assert "United States only" in jurisdictions
    assert "India" in jurisdictions


def test_model_anomaly_is_not_safety_certificate():
    result = search_knowledge("Does an Isolation Forest anomaly mean that water is unsafe?", top_k=4)
    assert result["found"] is True
    assert any("Model Interpretation" in source["title"] for source in result["sources"])


def test_blank_query_returns_helpful_result():
    result = search_knowledge("   ")
    assert result["found"] is False
    assert result["passages"] == []
