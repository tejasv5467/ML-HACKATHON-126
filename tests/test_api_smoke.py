"""Optional live smoke tests for a running student API.

Set PATHSNAP_API_URL, e.g. http://127.0.0.1:8002, before running pytest.
"""
import json
import os
from urllib.error import URLError
from urllib.request import urlopen

import pytest

BASE = os.getenv("PATHSNAP_API_URL")
pytestmark = pytest.mark.skipif(not BASE, reason="Set PATHSNAP_API_URL to test a running API")

def get_json(path):
    with urlopen(BASE.rstrip("/") + path, timeout=5) as response:
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))

def test_health_endpoint():
    payload = get_json("/")
    assert payload["status"] == "running"
    assert "student-test" in payload["mode"]

def test_trips_endpoint():
    payload = get_json("/trips")
    assert isinstance(payload["trips"], list)
    assert len(payload["trips"]) == len(set(payload["trips"]))
