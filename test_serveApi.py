import pytest
from serveApi import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_health(client):
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert response.data == b"API is alive"

if __name__ == "__main__":
    pytest.main()
