"""A simple centroid tracker for stable object IDs across frames."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared: int = 20, max_distance: int = 60) -> None:
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[int, int, int, int]] = OrderedDict()
        self.centroids: Dict[int, Tuple[int, int]] = OrderedDict()
        self.disappeared: Dict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, bbox: Tuple[int, int, int, int]) -> None:
        self.objects[self.next_object_id] = bbox
        self.centroids[self.next_object_id] = self._bbox_to_centroid(bbox)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.centroids[object_id]
        del self.disappeared[object_id]

    @staticmethod
    def _bbox_to_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return dict(self.objects)

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, rect in enumerate(rects):
            input_centroids[i] = self._bbox_to_centroid(rect)

        if len(self.objects) == 0:
            for rect in rects:
                self._register(rect)
            return dict(self.objects)

        object_ids = list(self.centroids.keys())
        object_centroids = np.array(list(self.centroids.values()))

        # Distance matrix: rows -> existing objects, cols -> new detections
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
            axis=2,
        )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = rects[col]
            self.centroids[object_id] = tuple(input_centroids[col])
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distances.shape[0])) - used_rows
        unused_cols = set(range(distances.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        for col in unused_cols:
            self._register(rects[col])

        return dict(self.objects)
