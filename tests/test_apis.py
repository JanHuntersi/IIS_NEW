import test_mbajk_api as mbajk_api
import test_weather_api as weather_api
import pytest


def main():
    mbajk_api()
    weather_api()

if __name__ == "__main__":
    pytest.main()
