import openmeteo_requests
import json
import requests_cache
from retry_requests import retry
from datetime import datetime


def get_forecast_data(latitude, longitude, station_name):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    # Initialize the Open-Meteo API client

    # Define the parameters for the weather data request
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                     "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"],
        "forecast_hours": 6,
    }

    # Make the API request
    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)

    response = responses[0]

    # Extract hourly data
    current = response.Hourly()
    
    hourly_data = {
        "station_number": station_name,
        "temperature": current.Variables(0).ValuesAsNumpy(),
        "relative_humidity": current.Variables(1).ValuesAsNumpy(),
        "dew_point": current.Variables(2).ValuesAsNumpy(),
        "apparent_temperature": current.Variables(3).ValuesAsNumpy(),
        "precipitation_probability": current.Variables(4).ValuesAsNumpy(),
        "rain": current.Variables(5).ValuesAsNumpy(),
        "surface_pressure": current.Variables(6).ValuesAsNumpy(),
    }

    return hourly_data
