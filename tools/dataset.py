#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This script fetches weather data from the Open-Meteo API for a given
# location and time range. The data is fetched hourly and can include the
# following features: temperature_2m, relative_humidity_2m, and rain.
#
# Usage:
#   python tools/dataset.py --location=Berlin \
#                            --start_date=2021-01-01 \
#                            --end_date=2021-01-02 \
#                            --features=temperature_2m,relative_humidity_2m,rain \
#                            --format=csv
#
# Arguments:
#   --location: The location for which the weather data should be fetched.
#   --start_date: The start date of the time range for the weather data.
#   --end_date: The end date of the time range for the weather data.
#   --features: The features to include in the weather data.
#   --format: The format of the weather data (csv, json, raw).
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

import pandas as pd
import requests
from pydantic import BaseModel, ValidationError, model_validator

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

    @model_validator(mode="before")
    def parse_features(values: dict):
        """Parse features from comma separated string."""
        if isinstance(values.get("features"), str):
            values["features"] = values.get("features").split(",")
        else:
            values["features"] = [
                Features.temperature_2m,
                Features.relative_humidity_2m,
                Features.rain,
            ]
        return values


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
        return pd.DataFrame()

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": settings.start_date.strftime("%Y-%m-%d"),
        "end_date": settings.end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join([v.value for v in settings.features]),
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


def undersampling_majority_class(
    df_majority: pd.DataFrame, df_minority: pd.DataFrame
) -> pd.DataFrame:
    # Undersample the majority class to match the minority class size
    df_majority_undersampled = df_majority.sample(len(df_minority), random_state=42)

    # Combine the minority class with the undersampled majority class
    df_balanced = pd.concat([df_minority, df_majority_undersampled])

    # Shuffle the resulting balanced dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    data["rain"] = data["rain"].apply(lambda x: 1 if x > 0 else 0)

    # In my case, the majority class is when it does not rain
    data = undersampling_majority_class(data[data["rain"] == 0], data[data["rain"] == 1])

    # Calculate z-score for temperature and humidity
    for col in ["temperature_2m", "relative_humidity_2m"]:
        data[f"{col}_zscore"] = (data[col] - data[col].mean()) / data[col].std(ddof=0)
    return data


def main():
    now = datetime.now(timezone.utc).date()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="The location for which the weather data should be fetched.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=now - timedelta(days=1),
        help="The start date of the time range for the weather data.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=now,
        help="The end date of the time range for the weather data.",
    )
    parser.add_argument(
        "--features",
        help="The features to include in the weather data.",
    )
    parser.add_argument(
        "--format", type=str, default="csv", help="The format of the weather data (csv, json, raw)."
    )

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
    data = transform_data(get_data(settings))
    if not data.empty:
        if settings.format == Format.csv:
            print(data.to_csv(index=False))
        elif settings.format == Format.json:
            print(data.to_json(orient="records", lines=True))
        else:
            print(data)


if __name__ == "__main__":
    main()
