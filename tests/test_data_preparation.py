"""Small synthetic checks for the real-data preparation contract.

These tests exercise code paths only; they do not create or replace project
results and do not require access to the organiser's data.
"""

import numpy as np
import pytest

from pathsnap_real_training import FEATURES, candidate_features, prepare_trip


def _small_trip(tmp_path):
    trip = tmp_path / "00000001"
    trip.mkdir()
    # Deliberately unsorted, with one out-of-range longitude and one NaN.
    (trip / "00000001.track").write_text(
        "-122.0000 37.0000 3\n"
        "181 37.0000 1\n"
        "-122.0002 37.0000 2\n"
        "nan 37.0000 4\n"
    )
    (trip / "00000001.nodes").write_text(
        "-122.0010 37.0000\n-122.0005 37.0000\n"
        "-122.0000 37.0000\n-121.9995 37.0000\n"
        "-121.9990 37.0000\n-121.9985 37.0000\n"
    )
    (trip / "00000001.arcs").write_text("0 1\n1 2\n2 3\n3 4\n4 5\n")
    (trip / "00000001.route").write_text("0\n1\n2\n3\n4\n")
    return trip


def test_prepare_trip_filters_invalid_gps_and_sorts_time(tmp_path):
    data, inventory = prepare_trip(_small_trip(tmp_path))

    assert inventory["raw_gps"] == 4
    assert inventory["valid_gps"] == 2
    assert data["X"].shape == (2, 5, len(FEATURES))
    assert np.all(np.isfinite(data["X"]))
    # The first sorted observation is at t=2, so its first time gap is zero.
    assert np.all(data["X"][0, :, FEATURES.index("time_gap_s")] == 0)
    assert np.all(data["X"][1, :, FEATURES.index("time_gap_s")] == 1)


def test_candidate_features_have_five_ranked_candidates_and_seven_features():
    track = np.array([[-122.0, 37.0, 10.0], [-121.9999, 37.0, 11.0]])
    nodes = np.array([
        [-122.001, 37.0], [-122.0005, 37.0], [-122.0, 37.0],
        [-121.9995, 37.0], [-121.999, 37.0], [-121.9985, 37.0],
    ])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])

    features, ids, _, _ = candidate_features(track, nodes, arcs)

    assert features.shape == (2, 5, 7)
    assert ids.shape == (2, 5)
    assert np.array_equal(features[0, :, FEATURES.index("candidate_rank")], np.arange(5))
    assert np.all(np.diff(features[0, :, FEATURES.index("distance_m")]) >= 0)
    assert np.isfinite(features).all()


def test_candidate_generation_rejects_fewer_than_five_arcs():
    track = np.array([[-122.0, 37.0, 10.0]])
    nodes = np.array([[-122.001, 37.0], [-122.0, 37.0], [-121.999, 37.0],
                       [-121.998, 37.0]])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [0, 2]])

    with pytest.raises(ValueError, match="Fewer road segments"):
        candidate_features(track, nodes, arcs)


def test_candidate_generation_rejects_nonfinite_feature_values():
    track = np.array([[np.nan, 37.0, 10.0]])
    nodes = np.array([[-122.001, 37.0], [-122.0, 37.0], [-121.999, 37.0],
                      [-121.998, 37.0], [-121.997, 37.0], [-121.996, 37.0]])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])

    with pytest.raises(ValueError, match="Non-finite candidate features"):
        candidate_features(track, nodes, arcs)
