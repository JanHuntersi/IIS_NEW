import requests
import pytest

weater_api_url = "https://api.open-meteo.com/v1/forecast"

def main():
    contract = "maribor"
    api_key = "5e150537116dbc1786ce5bec6975a8603286526b"
    url = f"https://api.jcdecaux.com/vls/v1/stations?contract={contract}&apiKey={api_key}"
    response = requests.get(url)
    print(response.status_code)
    assert response.status_code == 200

    res = requests.get(weater_api_url)
    print(res.status_code)
    assert res.status_code == 200

if __name__ == "__main__":
    pytest.main()
