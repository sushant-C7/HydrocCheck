from app.app import app
def test_rule_gate():
    c = app.test_client()
    r = c.post("/predict", json={"ph":3.0,"Hardness":1,"Solids":1,"Chloramines":1,"Sulfate":1,
        "Conductivity":1,"Organic_carbon":1,"Trihalomethanes":1,"Turbidity":1})
    assert r.get_json()["potable"] is False