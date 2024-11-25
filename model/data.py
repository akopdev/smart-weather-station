import logging
from datetime import date
from enum import Enum

import pandas as pd
import requests
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


class Features(str, Enum):
    temperature_2m = "temperature_2m"
    relative_humidity_2m = "relative_humidity_2m"
    rain = "rain"


class Location(BaseModel):
    """Response from the Open-Meteo API."""

    latitude: float
    longitude: float


class DataProvider:
    """
    A class to fetch weather data.

    It is using free tenant from the Open-Meteo API. Not API key is required.
    You can reimplement this class to use a provider of your choice.

    Example:
        provider = DataProvider("Berlin")
        data = provider.get_data(date(2021, 1, 1), date(2021, 2, 1))
    """
    def __init__(self, location: str):
        self.location = location

    def get_coordinates(self) -> Location:
        """Get the latitude and longitude of a city."""
        with requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": self.location, "count": 1},
        ) as response:
            data = response.json()
            if not data.get("results"):
                log.error("No location found for %s", self.location)
                return
            try:
                return Location(**data.get("results", [{}])[0])
            except ValidationError as e:
                log.error("Error while parsing location data: %s", e)
                return

    def get_data(self, start_date: date, end_date: date = date.today()) -> pd.DataFrame:
        """Get the weather data from the Open-Meteo API.

        The data is fetched hourly for the given location and time range.

        Parameters:
        ----------
            settings: The settings for the data fetch.

        Returns:
        -------
            The weather data from the Open-Meteo API.
        """
        location = self.get_coordinates()
        if not location:
            return pd.DataFrame()

        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,relative_humidity_2m,rain",
        }

        with requests.get(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        ) as response:
            if response.status_code != 200:
                log.error("Error while fetching data: %s", response.text)
                return
            data = response.json()
            if not data.get("hourly"):
                log.error("No data found for %s", self.location)
                return
            return pd.DataFrame(data.get("hourly"))
