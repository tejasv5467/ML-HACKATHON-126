import numpy as np
from pathsnap_real_hmm import decode, collapse, metrics

def test_empty_sequence():
    assert decode(np.empty((0, 2), dtype=int), np.empty((0, 2)),
                  np.empty(0), np.array([[0, 1], [1, 2]] )).size == 0

def test_directed_transition_prefers_connected_arc():
    # At t=0, arc 0 has higher emission; at t=1 arc 2 has higher emission.
    # Arc 0 ends at node 1; arc 1 starts at node 1, while arc 2 is disconnected.
    # With beta=0.5 and penalty=12, Viterbi should choose arc 1.
    ids = np.array([[0, 2], [2, 1]], dtype=int)
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    gaps = np.array([0.0, 1.0])
    arcs = np.array([[0, 1], [1, 2], [8, 9]], dtype=int)
    out = decode(ids, probs, gaps, arcs)
    assert out.tolist() == [0, 1]

def test_gap_resets_sequence():
    # A long gap splits the sequence; the second observation should select its
    # locally strongest candidate, even though it is disconnected.
    ids = np.array([[0, 1], [2, 3]], dtype=int)
    probs = np.array([[0.99, 0.01], [0.01, 0.99]])
    gaps = np.array([0.0, 61.0])
    arcs = np.array([[0, 1], [1, 2], [8, 9], [9, 10]], dtype=int)
    out = decode(ids, probs, gaps, arcs)
    assert out.tolist() == [0, 3]

def test_nonpositive_gap_resets_sequence():
    ids = np.array([[0, 1], [2, 3]], dtype=int)
    probs = np.array([[0.99, 0.01], [0.01, 0.99]])
    gaps = np.array([0.0, 0.0])
    arcs = np.array([[0, 1], [1, 2], [8, 9], [9, 10]], dtype=int)
    assert decode(ids, probs, gaps, arcs).tolist() == [0, 3]

def test_backtracking_and_collapse():
    ids = np.array([[0, 1], [2, 3], [4, 5]], dtype=int)
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    gaps = np.array([0.0, 1.0, 1.0])
    # 0 -> 2 -> 4 is a directed continuous route.
    arcs = np.array([[0, 1], [10, 11], [1, 2], [20, 21], [2, 3], [30, 31]], dtype=int)
    out = decode(ids, probs, gaps, arcs)
    assert out.tolist() == [0, 2, 4]
    assert collapse([0, 0, 2, 2, 4]) == [0, 2, 4]

def test_decoder_rejects_invalid_probabilities():
    ids = np.array([[0, 1]], dtype=int)
    probs = np.array([[np.nan, 0.2]])
    with np.testing.assert_raises(AssertionError):
        decode(ids, probs, np.array([0.0]), np.array([[0,1],[1,2]]))
