"""SORT-style Kalman + Hungarian point tracker for KuRALSNetNPUSeg's
per-pixel segmentation output.

Pipeline: connected-components centroid extraction per frame (scipy.ndimage,
same continuous-centroid convention kurals/loaders/dataloaders_detect.py's
encode_detection_target already uses -- mean of a blob's pixel coordinates)
-> constant-velocity Kalman filter per track -> scipy.optimize's Hungarian
assignment for frame-to-frame association -> standard SORT track lifecycle
(tentative -> confirmed -> deleted).

No physical frame-rate/timestamp constant exists anywhere in this codebase
(confirmed via grep across kurals/dataset_process and kurals/loaders) -- dt is
always exactly 1 frame here, so velocity state and every noise/gating
constant below are in frame/pixel units, not physical units (m, m/s). Tune
max_match_dist/process_var/measurement_var empirically against real
sequences rather than deriving them analytically.
"""
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.optimize import linear_sum_assignment


def extract_centroids(class_mask):
    """class_mask: 2D bool array, True where the target class is predicted.

    Returns a list of (y, x) float centroids, one per 4-connected blob --
    same "mean of the blob's pixel coordinates" convention as
    encode_detection_target's ys.mean()/xs.mean() (dataloaders_detect.py)."""
    labeled, n = label(class_mask)
    if n == 0:
        return []
    # center_of_mass always returns a list when index is a sequence (even of
    # length 1) -- range(1, n + 1) is a sequence, so no extra wrapping needed.
    coms = center_of_mass(class_mask, labeled, range(1, n + 1))
    return [(float(cy), float(cx)) for cy, cx in coms]


class KalmanPoint:
    """Constant-velocity Kalman filter, state [x, y, vx, vy], dt=1 frame."""

    def __init__(self, y, x, process_var=1.0, measurement_var=4.0):
        self.state = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 10.0
        self.F = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * measurement_var

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[0], self.state[1]  # predicted (x, y)

    def update(self, y, x):
        z = np.array([x, y])
        innovation = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self):
        """Current (y, x) position estimate."""
        return float(self.state[1]), float(self.state[0])


class Track:
    """One tracked target: a Kalman filter plus SORT lifecycle bookkeeping."""

    _next_id = 1

    def __init__(self, y, x, process_var, measurement_var):
        self.kf = KalmanPoint(y, x, process_var, measurement_var)
        self.id = Track._next_id
        Track._next_id += 1
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.history = [(y, x)]

    def predict(self):
        return self.kf.predict()  # (x, y)

    def update(self, y, x):
        self.kf.update(y, x)
        self.hits += 1
        self.misses = 0
        self.history.append(self.kf.position)

    def mark_missed(self):
        self.misses += 1


class SortTracker:
    """SORT-style multi-target tracker over per-frame (y, x) centroids.

    PARAMETERS
    ----------
    max_match_dist: float
        Gating distance in pixels -- a detection farther than this from
        every predicted track position is never matched (a new track spawns
        instead). Tune against the dataset's real inter-frame motion; no
        principled value derivable without physical units (see module
        docstring).
    confirm_hits: int
        Consecutive matched frames before a tentative track is reported as
        confirmed -- suppresses single-frame detection noise from spawning
        a spurious track.
    max_misses: int
        Consecutive missed frames before a track is deleted.
    process_var, measurement_var: float
        Kalman filter process/measurement noise (frame/pixel units).
    """

    def __init__(self, max_match_dist=15.0, confirm_hits=3, max_misses=5,
                 process_var=1.0, measurement_var=4.0):
        self.max_match_dist = max_match_dist
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.tracks = []

    def step(self, detections):
        """detections: list of (y, x) centroids for the current frame (see
        extract_centroids). Returns the list of currently-confirmed Track
        objects, after this frame's predict/associate/update/lifecycle pass."""
        predicted_xy = [t.predict() for t in self.tracks]  # each (x, y)

        matched_tracks, matched_dets = set(), set()
        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, (px, py) in enumerate(predicted_xy):
                for j, (dy, dx) in enumerate(detections):
                    cost[i, j] = np.hypot(px - dx, py - dy)
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= self.max_match_dist:
                    self.tracks[r].update(*detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.mark_missed()

        for j, (y, x) in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(y, x, self.process_var, self.measurement_var))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for t in self.tracks:
            if t.hits >= self.confirm_hits:
                t.confirmed = True

        return [t for t in self.tracks if t.confirmed]
