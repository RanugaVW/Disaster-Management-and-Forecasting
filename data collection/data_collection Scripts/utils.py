"""
utils.py
--------
Shared utilities for Open-Meteo API calls.

Features:
  - Configurable date range and base URL
  - Exponential back-off retry (up to 5 attempts)
  - Polite rate-limiting between requests
  - JSON-based progress tracker so collection can resume after interruption
"""

import json
import os
import time
import logging
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Global Config
# ---------------------------------------------------------------------------
BASE_URL   = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2015-01-01"
END_DATE   = "2026-01-01"
TIMEZONE   = "Asia/Colombo"

# Time (seconds) to sleep between API calls — stay well within free-tier limits
REQUEST_DELAY = 1.0          # ~60 req/min upper limit; 1.0 s ensures we stay steady
MAX_RETRIES   = 10

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openmeteo")


# ---------------------------------------------------------------------------
# Core API fetcher
# ---------------------------------------------------------------------------
def fetch_data(
    lat: float,
    lon: float,
    daily: list | None = None,
    hourly: list | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict | None:
    """
    Fetch data from the Open-Meteo historical archive API.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates of the target location.
    daily : list[str], optional
        Daily variable names (e.g. ["temperature_2m_mean", "rain_sum"]).
    hourly : list[str], optional
        Hourly variable names (e.g. ["soil_moisture_7_to_28cm"]).
    start_date, end_date : str, optional
        Y-M-D strings. If None, uses defaults from utils.py.

    Returns
    -------
    dict or None
        Parsed JSON response, or None if all retry attempts fail.
    """
    params: dict = {
        "latitude":           lat,
        "longitude":          lon,
        "start_date":         start_date or START_DATE,
        "end_date":           end_date or END_DATE,
        "timezone":           TIMEZONE,
        "temporal_resolution": "native",
    }
    if daily:
        params["daily"]  = ",".join(daily)
    if hourly:
        params["hourly"] = ",".join(hourly)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            log.warning(f"  HTTP {resp.status_code} on attempt {attempt}/{MAX_RETRIES}: {e}")
            if resp.status_code == 429:
                log.warning("  -> Hit Open-Meteo API burst limit. Cooling down for 60 seconds...")
                time.sleep(60)
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout on attempt {attempt}/{MAX_RETRIES}")
        except requests.exceptions.RequestException as e:
            log.warning(f"  Request error on attempt {attempt}/{MAX_RETRIES}: {e}")

        if attempt < MAX_RETRIES:
            wait = min(300, 4 ** attempt)  # Cap maximum backoff at 5 minutes
            log.info(f"  Retrying in {wait}s …")
            time.sleep(wait)

    log.error(f"  FAILED after {MAX_RETRIES} attempts for lat={lat}, lon={lon}")
    return None


def rate_limit():
    """Sleep between API calls to avoid throttling."""
    time.sleep(REQUEST_DELAY)


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------
class ProgressTracker:
    """
    Lightweight JSON-backed progress store.

    Keeps a set of completed DS IDs per task so interrupted runs can resume.

    Example
    -------
    tracker = ProgressTracker("progress.json", task="temperature")
    if tracker.is_done("Colombo"):
        continue
    # … fetch …
    tracker.mark_done("Colombo")
    """

    def __init__(self, filepath: str | Path, task: str):
        self.filepath = Path(filepath)
        self.task     = task
        self._data: dict = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                self._data = json.load(f)
        if self.task not in self._data:
            self._data[self.task] = []

    def _save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    def is_done(self, ds_id: str) -> bool:
        return ds_id in self._data[self.task]

    def mark_done(self, ds_id: str):
        if ds_id not in self._data[self.task]:
            self._data[self.task].append(ds_id)
            self._save()

    def completed_count(self) -> int:
        return len(self._data[self.task])

    def reset(self):
        self._data[self.task] = []
        self._save()
        log.info(f"Progress for '{self.task}' has been reset.")
