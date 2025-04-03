import pytest
from fastapi.testclient import TestClient
from main import app, DEV_API_KEY

client = TestClient(app)
HEADERS = {"X-API-Key": DEV_API_KEY}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data and data["api"] == "Cricket Match Analysis API"

def test_get_available_teams():
    response = client.get("/teams", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "teams" in data
    assert isinstance(data["teams"], list)

def test_get_match_summary():
    response = client.get("/summary?team1=India&team2=Australia", headers=HEADERS)
    assert response.status_code in [200, 404]  # Accepts both valid and no data cases
    if response.status_code == 200:
        data = response.json()
        assert "total_matches" in data
        assert isinstance(data["total_matches"], int)

def test_get_match_summary_invalid_teams():
    response = client.get("/summary?team1=FakeTeam&team2=NonExistentTeam", headers=HEADERS)
    assert response.status_code == 404

def test_clear_cache():
    response = client.get("/clear_cache", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data.get("message") == "Cache cleared and data reloaded"

def test_unauthorized_access():
    response = client.get("/summary?team1=India&team2=Australia")
    assert response.status_code == 401
