import math


class Tracker:
    """
    Greedy nearest-neighbour association so IDs stay stable under faster motion.
    `max_match_dist` should be a fair fraction of frame size (e.g. 70–90 px at 1020-wide).
    """

    def __init__(self, max_match_dist=78):
        self.center_points = {}
        self.id_count = 0
        self.max_match_dist = max_match_dist

    def update(self, objects_rect):
        objects_bbs_ids = []
        if not objects_rect:
            self.center_points = {}
            return objects_bbs_ids

        det_centers = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            det_centers.append((cx, cy))

        track_ids = list(self.center_points.keys())
        pairs = []
        for di, (dcx, dcy) in enumerate(det_centers):
            for tid in track_ids:
                tcx, tcy = self.center_points[tid]
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if dist < self.max_match_dist:
                    pairs.append((dist, di, tid))

        pairs.sort(key=lambda t: t[0])
        assigned_det = set()
        assigned_tid = set()

        for dist, di, tid in pairs:
            if di in assigned_det or tid in assigned_tid:
                continue
            assigned_det.add(di)
            assigned_tid.add(tid)
            rect = objects_rect[di]
            x, y, w, h = rect
            cx, cy = det_centers[di]
            self.center_points[tid] = (cx, cy)
            objects_bbs_ids.append([x, y, w, h, tid])

        for di, rect in enumerate(objects_rect):
            if di in assigned_det:
                continue
            x, y, w, h = rect
            cx, cy = det_centers[di]
            tid = self.id_count
            self.id_count += 1
            self.center_points[tid] = (cx, cy)
            objects_bbs_ids.append([x, y, w, h, tid])

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            new_center_points[object_id] = self.center_points[object_id]
        self.center_points = new_center_points
        return objects_bbs_ids
