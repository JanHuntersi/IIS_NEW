import requests

weater_api_url = "https://api.open-meteo.com/v1/forecast"

def weather_api():

    res = requests.get(weater_api_url)
    print(res.status_code)
    assert res.status_code == 200

weather_api()    
