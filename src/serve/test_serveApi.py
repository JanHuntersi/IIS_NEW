import pytest
from src.serve.test_serveApi import app
from unittest.mock import patch

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@patch("serveApi.joblib.load")
def test_health(mock_load, client):
    # Mocking the joblib.load function
    mock_load.return_value = None  # or whatever value you expect

    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert response.data == b"API is alive"

if __name__ == "__main__":
    pytest.main()
