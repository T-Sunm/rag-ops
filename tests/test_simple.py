import os
from fastapi.testclient import TestClient

# Test environment setup
os.environ["ENVIRONMENT"] = "test"

from src.main import app
from src.services.application.rag import rag_service
from nemoguardrails import LLMRails, RailsConfig


# Initialize app state for testing
def setup_app_state():
    """Setup application state for testing"""
    app.state.rag_service = rag_service

    config_restapi = RailsConfig.from_path("guardrails/config_restapi")
    rails_restapi = LLMRails(config_restapi)
    app.state.rails_restapi = rails_restapi

    config_sse = RailsConfig.from_path("guardrails/config_sse")
    rails_sse = LLMRails(config_sse)
    app.state.rails_sse = rails_sse


# Setup app state before creating client
setup_app_state()
client = TestClient(app)


def test_settings_import():
    """Test có thể import settings"""
    from src.config.settings import SETTINGS

    assert SETTINGS.ENVIRONMENT == "test"
    assert SETTINGS.HOST == "0.0.0.0"
    assert SETTINGS.PORT == 8000


def test_app_creation():

    from src.main import app

    assert app is not None
    assert app.title == "RAG Ops - Production Architecture"


def test_health_endpoint():
    """Test /health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint():
    """Test /ready endpoint"""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_question():
    """Test asking a simple question to the RAG endpoint"""
    response = client.post(
        "/v1/rest-retrieve/",
        json={
            "user_input": "What is battery recycling?",
            "session_id": "test_session_123",
            "user_id": "test_user_456",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "session_id" in data
    assert "user_id" in data
    assert data["session_id"] == "test_session_123"
    assert data["user_id"] == "test_user_456"
    assert len(data["response"]) > 0
