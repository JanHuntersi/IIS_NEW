from test_mbajk_api import mbajk_api
from test_weather_api import weather_api
import pytest


def main():
    mbajk_api()
    weather_api()

if __name__ == "__main__":
    pytest.main()
