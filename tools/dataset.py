import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

import pandas as pd
import requests
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


class Features(str, Enum):
    temperature_2m = "temperature_2m"
    relative_humidity_2m = "relative_humidity_2m"
    rain = "rain"


class Format(str, Enum):
    csv = "csv"
    json = "json"
    raw = "raw"


class Settings(BaseModel):
    location: str
    start_date: date
    end_date: date
    features: Optional[List[Features]] = []
    format: Optional[Format] = Format.csv


class Location(BaseModel):
    """Response from the Open-Meteo API."""

    latitude: float
    longitude: float


def get_location(name: str) -> Location:
    """Get the latitude and longitude of a city."""
    with requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1},
    ) as response:
        data = response.json()
        if not data.get("results"):
            log.error("No location found for %s", name)
            return
        try:
            return Location(**data.get("results", [{}])[0])
        except ValidationError as e:
            log.error("Error while parsing location data: %s", e)
            return


def get_data(settings: Settings) -> pd.DataFrame:
    """Get the weather data from the Open-Meteo API.

    The data is fetched hourly for the given location and time range.

    Parameters:
    ----------
        settings: The settings for the data fetch.

    Returns:
    -------
        The weather data from the Open-Meteo API.
    """
    location = get_location(settings.location)
    if not location:
        return

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": settings.start_date.strftime("%Y-%m-%d"),
        "end_date": settings.end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(settings.features),
    }

    with requests.get("https://archive-api.open-meteo.com/v1/archive", params=params) as response:
        if response.status_code != 200:
            log.error("Error while fetching data: %s", response.text)
            return
        data = response.json()
        if not data.get("hourly"):
            log.error("No data found for %s", settings.location)
            return
        return pd.DataFrame(data.get("hourly"))


def main():
    now = datetime.now(timezone.utc).date()

    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--start_date", type=str, default=now - timedelta(days=1))
    parser.add_argument("--end_date", type=str, default=now)
    parser.add_argument(
        "--features", type=list, default=["temperature_2m", "relative_humidity_2m", "rain"]
    )
    parser.add_argument("--format", type=str, default="csv")

    try:
        args = parser.parse_args()
        settings = Settings(**vars(args))
    except ValidationError as e:
        error = e.errors(include_url=False, include_context=False)[0]
        sys.exit(
            "Wrong argument value passed ({}): {}".format(
                error.get("loc", ("system",))[0], error.get("msg")
            )
        )
    data = get_data(settings)
    if not data.empty:
        if settings.format == Format.csv:
            print(data.to_csv(index=False))
        elif settings.format == Format.json:
            print(data.to_json(orient="records", lines=True))
        else:
            print(data)


if __name__ == "__main__":
    main()
