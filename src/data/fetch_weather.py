import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime

def fetch_weather_data(lat=52.52, long=13.41,timestamp=None,dateStart=None,dateEnd=None):

    print(f"Fetching weather data for latitude {lat} and longitude {long}")

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"

    params = {}

    if dateStart is not None:
        params = {
        "latitude": lat,
        "longitude": long,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "precipitation", "surface_pressure"],
        "start_date":dateStart,
        "end_date":dateEnd,
        "timezone": "auto"
    }
    else:
        params = {
        "latitude": lat,
        "longitude": long,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "precipitation", "surface_pressure"],
        "forecast_days":2,
        "timezone": "auto"
    }

    
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
    hourly_precipitation_probability = hourly.Variables(4).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(5).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["dew_point_2m"] = hourly_dew_point_2m
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["precipitation_probability"] = hourly_precipitation_probability
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["surface_pressure"] = hourly_surface_pressure

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    if timestamp is None:
        
        return hourly_dataframe
    else:
       #Find station based on timestamp
        station_row = hourly_dataframe.loc[hourly_dataframe['date'] == timestamp].copy()

        #update timestamp 
        station_row['date'] = pd.to_datetime(station_row['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        #return station row
        return station_row
    
    

def get_forecast_data(latitude, longitude, station_number):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                     "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"],
        "forecast_hours": 6,
    }


    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]

    # Extract hourly data
    current = response.Hourly()
    hourly_data = {
        "station_number": station_number,
        "temperature_2m": current.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": current.Variables(1).ValuesAsNumpy(),
        "dew_point_2m": current.Variables(2).ValuesAsNumpy(),
        "apparent_temperature": current.Variables(3).ValuesAsNumpy(),
        "precipitation_probability": current.Variables(4).ValuesAsNumpy(),
        "rain": current.Variables(5).ValuesAsNumpy(),
        "surface_pressure": current.Variables(6).ValuesAsNumpy(),
    }

    return hourly_data
