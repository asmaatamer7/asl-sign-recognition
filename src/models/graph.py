"""Skeleton graph for the 49 kept joints (7 body + 21 left hand + 21 right hand)."""
import numpy as np

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
# body order: 0 nose, 1 L-shoulder, 2 R-shoulder, 3 L-elbow, 4 R-elbow, 5 L-wrist, 6 R-wrist
BODY_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]


def skeleton_edges():
    e = list(BODY_EDGES)
    e += [(7 + a, 7 + b) for a, b in HAND_EDGES]
    e += [(28 + a, 28 + b) for a, b in HAND_EDGES]
    e += [(5, 7), (6, 28)]                      # wrist <-> hand root
    return e


def adjacency(J=49):
    """Returns A (2, J, J): [identity, symmetric-normalised adjacency without self loops]."""
    A = np.zeros((J, J), np.float32)
    for a, b in skeleton_edges():
        A[a, b] = A[b, a] = 1
    d = A.sum(1); d[d == 0] = 1
    An = A / np.sqrt(d[:, None] * d[None, :])
    return np.stack([np.eye(J, dtype=np.float32), An])
