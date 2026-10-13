def rule_gate(s: dict):
    reasons = []
    if s.get("ph") is not None and s["ph"] < 4.0:
        reasons.append("pH below 4.0 (critical)")
    if s.get("Turbidity") is not None and s["Turbidity"] > 15:
        reasons.append("Turbidity above 15 NTU (critical)")
    return reasons