import requests

def mbajk_api():
    contract = "maribor"
    api_key = "5e150537116dbc1786ce5bec6975a8603286526b"
    url = f"https://api.jcdecaux.com/vls/v1/stations?contract={contract}&apiKey={api_key}"
    response = requests.get(url)
    print(response.status_code)
    assert response.status_code == 200
mbajk_api()
