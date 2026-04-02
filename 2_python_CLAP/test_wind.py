import config
import requests
import json

OPENWEATHER_API_KEY = config.openweather_api_key
WIND_LAT = 52.372
WIND_LON = 4.917

print('start')

def get_current_wind_speed():
    """Fetch the current wind speed from OpenWeather Current Weather API."""
    if not OPENWEATHER_API_KEY:
        raise ValueError("OpenWeather API key is not configured.")

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "lat": WIND_LAT,
            "lon": WIND_LON,
            "appid": OPENWEATHER_API_KEY,
        },
        timeout=10,
    )
    response.raise_for_status()

    weather_data = response.json()
    print(f"Fetched wind data: {weather_data['wind']['speed']} m/s")
    return weather_data["wind"]["speed"]

get_current_wind_speed()