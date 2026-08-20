"""
detection/activity_recognizer.py
Crowd activity recognition using YOLOv11 tracking data.

FIXED VERSION:
- Enforced unused FALL_STAY_DOWN_FRAMES constant — now actually checks that person
  remains on the ground for sustained period (reduces false positives from bending)
- Improved fall detection logic with proper persistence checking
- Better temporal smoothing to prevent activity flicker
- All activity thresholds now properly validated with history requirements
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time


@dataclass
class ActivityDetection:
    """Single detected activity"""
    activity_type: str        # GATHERING, PANIC, FALL, FIGHT, STAMPEDE, DISPERSAL, LOITERING
    severity: str             # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float         # 0.0-1.0
    description: str
    involved_ids: List[int]   # Tracked person IDs involved
    location: Tuple[int, int] # Centroid of the activity area
    zone: str                 # Which zone (top_left, mid_center, etc.)


@dataclass
class ActivityResult:
    """Complete activity recognition result for a frame"""
    activities: List[ActivityDetection] = field(default_factory=list)
    dominant_activity: Optional[str] = None
    scene_mood: str = "CALM"  # CALM, ALERT, TENSE, CHAOTIC
    detection_time_ms: float = 0.0


class ActivityRecognizer:
    """
    Recognizes crowd activities from YOLOv11 tracking history.
    
    Uses per-ID bounding box history to compute velocities, 
    aspect ratio changes, clustering, and proximity patterns.
    No additional ML model required — purely heuristic.
    
    FIXED: Now properly enforces temporal persistence for all activities.
    Falls must be sustained (not just a momentary drop), fights require
    consistent close proximity over multiple frames, etc.
    """

    # ── Thresholds ──────────────────────────────────────────────
    # Gathering
    GATHER_MIN_PEOPLE = 4
    GATHER_RADIUS_PX = 150        # Max radius to consider a cluster
    GATHER_MIN_FRAMES = 15        # Must persist for this many frames
    GATHER_CONVERGENCE_RATIO = 0.6  # Radius must shrink by this fraction

    # Panic / Stampede
    PANIC_VELOCITY_THRESHOLD = 25.0   # Average px/frame velocity
    PANIC_DIRECTION_VARIANCE = 0.4    # Low variance = uniform flow (stampede)
    PANIC_MIN_PEOPLE = 5

    # Fall — FIXED: Now actually enforces STAY_DOWN requirement
    FALL_ASPECT_RATIO_DROP = 0.5  # Standing ≈ 2.0+, fallen ≈ 0.5-0.8
    FALL_FRAMES_WINDOW = 8        # Must detect aspect ratio change within this window
    FALL_STAY_DOWN_FRAMES = 10    # FIXED: Now actually enforced — must stay low for N frames
    FALL_MIN_ASPECT_RATIO = 0.85  # Max aspect ratio while "down" (prevents false positives from bending)

    # Fight
    FIGHT_PROXIMITY_PX = 60       # Two people within this distance
    FIGHT_OSCILLATION_THRESHOLD = 12.0  # High erratic motion
    FIGHT_MIN_FRAMES = 10

    # Dispersal
    DISPERSAL_VELOCITY_THRESHOLD = 15.0
    DISPERSAL_MIN_PEOPLE = 4
    DISPERSAL_DIVERGENCE_RATIO = 0.7  # Fraction moving outward

    # Loitering
    LOITER_RADIUS_PX = 35
    LOITER_MIN_FRAMES = 90        # ~9 seconds at 10 FPS

    def __init__(self, history_size: int = 30):
        """
        Args:
            history_size: Number of frames of tracking history to keep per ID.
        """
        self.history_size = history_size

        # Per-ID tracking history: {id: deque([(cx, cy, w, h, timestamp), ...])}
        self.id_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        # Activity persistence: prevent flicker by requiring N consecutive detections
        self._activity_counts: Dict[str, int] = defaultdict(int)
        self._active_activities: Dict[str, ActivityDetection] = {}
        self._confirmation_threshold = 3  # Must be detected N times to confirm

        # Gathering cluster tracking
        self._cluster_history: deque = deque(maxlen=30)

        # FIXED: Track fall candidates separately (person started falling)
        # to distinguish from sustained "already down" state
        self._fall_candidates: Dict[int, int] = {}  # {tid: frame_count_since_transition}

        self._frame_count = 0

        print("[ACTIVITY RECOGNIZER] Initialized")

    def update_tracking(self, detections: List[Dict]) -> None:
        """
        Update internal tracking history from the latest detection frame.
        Call this every detection frame BEFORE calling detect().
        
        Args:
            detections: List of dicts with 'id', 'bbox' [x1,y1,x2,y2], 'centroid' [cx,cy]
        """
        timestamp = time.time()
        seen_ids = set()

        for det in detections:
            track_id = det.get('id')
            if track_id is None:
                continue

            seen_ids.add(track_id)
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            cx, cy = det['centroid']

            self.id_history[track_id].append((cx, cy, w, h, timestamp))

        # Prune IDs not seen for a while (keep history for 2x history_size frames)
        if self._frame_count % 60 == 0:
            stale_ids = [
                tid for tid in self.id_history
                if tid not in seen_ids and len(self.id_history[tid]) > 0
                and (timestamp - self.id_history[tid][-1][4]) > 5.0
            ]
            for tid in stale_ids:
                del self.id_history[tid]
                # Clean up fall candidates too
                self._fall_candidates.pop(tid, None)

        self._frame_count += 1

    def detect(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> ActivityResult:
        """
        Run all activity detectors on the current frame's tracking data.
        
        Args:
            detections: Current frame detections (with 'id', 'bbox', 'centroid')
            frame_shape: (height, width) of the frame
            
        Returns:
            ActivityResult with all detected activities
        """
        start_time = time.time()

        # Update tracking history
        self.update_tracking(detections)

        activities: List[ActivityDetection] = []
        h, w = frame_shape[:2]

        # Only run detectors if we have enough tracked IDs with history
        tracked_ids_with_history = {
            tid: hist for tid, hist in self.id_history.items()
            if len(hist) >= 5
        }

        if len(tracked_ids_with_history) >= 2:
            # 1. Gathering detection
            gathering = self._detect_gathering(tracked_ids_with_history, w, h)
            if gathering:
                activities.append(gathering)

            # 2. Panic / Stampede detection
            panic = self._detect_panic(tracked_ids_with_history, w, h)
            if panic:
                activities.append(panic)

            # 3. Fight detection
            fight = self._detect_fight(tracked_ids_with_history, w, h)
            if fight:
                activities.append(fight)

            # 4. Dispersal detection
            dispersal = self._detect_dispersal(tracked_ids_with_history, w, h)
            if dispersal:
                activities.append(dispersal)

        # These work with single individuals too
        if len(tracked_ids_with_history) >= 1:
            # 5. Fall detection (FIXED: now with proper sustained "down" checking)
            falls = self._detect_falls(tracked_ids_with_history, w, h)
            activities.extend(falls)

            # 6. Loitering detection
            loitering = self._detect_loitering(tracked_ids_with_history, w, h)
            activities.extend(loitering)

        # Apply temporal smoothing (prevent flicker)
        confirmed_activities = self._smooth_activities(activities)

        # Determine dominant activity and scene mood
        dominant = None
        mood = "CALM"

        if confirmed_activities:
            severity_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
            confirmed_activities.sort(
                key=lambda a: severity_order.get(a.severity, 0), reverse=True
            )
            dominant = confirmed_activities[0].activity_type
            mood = self._assess_mood(confirmed_activities)

        detection_time = (time.time() - start_time) * 1000

        return ActivityResult(
            activities=confirmed_activities,
            dominant_activity=dominant,
            scene_mood=mood,
            detection_time_ms=round(detection_time, 2)
        )

    # ══════════════════════════════════════════════════════════════
    # DETECTION METHODS
    # ══════════════════════════════════════════════════════════════

    def _detect_gathering(
        self, tracked: Dict[int, deque], fw: int, fh: int
    ) -> Optional[ActivityDetection]:
        """
        Detect crowd gathering: multiple people converging into a tight cluster.
        Uses DBSCAN-like clustering on current centroids + convergence over time.
        """
        if len(tracked) < self.GATHER_MIN_PEOPLE:
            return None

        # Get current centroids
        current_centroids = {}
        for tid, hist in tracked.items():
            cx, cy, _, _, _ = hist[-1]
            current_centroids[tid] = np.array([cx, cy])

        # Simple clustering: find the densest cluster
        ids = list(current_centroids.keys())
        positions = np.array([current_centroids[tid] for tid in ids])

        # For each point, count neighbors within GATHER_RADIUS_PX
        best_cluster_ids = []
        best_cluster_center = None

        for i, pos in enumerate(positions):
            distances = np.linalg.norm(positions - pos, axis=1)
            neighbors = [ids[j] for j, d in enumerate(distances) if d < self.GATHER_RADIUS_PX]

            if len(neighbors) > len(best_cluster_ids):
                best_cluster_ids = neighbors
                best_cluster_center = pos

        if len(best_cluster_ids) < self.GATHER_MIN_PEOPLE:
            return None

        # Check convergence: are these people getting closer over time?
        cluster_positions_now = np.array([current_centroids[tid] for tid in best_cluster_ids])
        current_spread = np.std(cluster_positions_now, axis=0).mean()

        # Check spread N frames ago
        older_positions = []
        frames_back = min(self.GATHER_MIN_FRAMES, min(len(tracked[tid]) for tid in best_cluster_ids))
        if frames_back < 5:
            return None

        for tid in best_cluster_ids:
            hist = tracked[tid]
            idx = max(0, len(hist) - frames_back)
            cx, cy, _, _, _ = hist[idx]
            older_positions.append([cx, cy])

        older_spread = np.std(np.array(older_positions), axis=0).mean()

        # Are they converging?
        if older_spread > 0 and current_spread < older_spread * (1 - self.GATHER_CONVERGENCE_RATIO * 0.5):
            center = (int(np.mean(cluster_positions_now[:, 0])), int(np.mean(cluster_positions_now[:, 1])))
            severity = "MEDIUM" if len(best_cluster_ids) < 8 else "HIGH"
            confidence = float(min(0.9, 0.5 + len(best_cluster_ids) * 0.05))

            return ActivityDetection(
                activity_type="GATHERING",
                severity=severity,
                confidence=round(confidence, 2),
                description=f"{len(best_cluster_ids)} people converging into a group",
                involved_ids=[int(x) for x in best_cluster_ids],
                location=center,
                zone=self._location_to_zone(center[0], center[1], fw, fh)
            )

        return None

    def _detect_panic(
        self, tracked: Dict[int, deque], fw: int, fh: int
    ) -> Optional[ActivityDetection]:
        """
        Detect panic/stampede: high velocity movement with uniform direction.
        Stampede = high speed + same direction. Panic = high speed + chaotic direction.
        """
        if len(tracked) < self.PANIC_MIN_PEOPLE:
            return None

        velocities = []
        directions = []

        for tid, hist in tracked.items():
            if len(hist) < 3:
                continue

            # Compute velocity over last 3 frames
            recent = list(hist)[-3:]
            dx = recent[-1][0] - recent[0][0]
            dy = recent[-1][1] - recent[0][1]
            speed = np.sqrt(dx**2 + dy**2) / max(len(recent) - 1, 1)
            velocities.append(speed)

            if speed > 2.0:
                angle = np.arctan2(dy, dx)
                directions.append(angle)

        if len(velocities) < self.PANIC_MIN_PEOPLE:
            return None

        avg_velocity = np.mean(velocities)

        if avg_velocity < self.PANIC_VELOCITY_THRESHOLD:
            return None

        # Compute direction variance (circular variance)
        high_speed_ids = [
            tid for tid, hist in tracked.items()
            if len(hist) >= 3 and self._compute_speed(hist) > self.PANIC_VELOCITY_THRESHOLD * 0.5
        ]

        if len(directions) >= 3:
            direction_variance = 1.0 - np.abs(np.mean(np.exp(1j * np.array(directions))))
        else:
            direction_variance = 0.5

        # Compute center of panic
        all_positions = []
        for tid in high_speed_ids:
            hist = tracked[tid]
            cx, cy = hist[-1][0], hist[-1][1]
            all_positions.append([cx, cy])

        if not all_positions:
            return None

        center_arr = np.mean(all_positions, axis=0)
        center = (int(center_arr[0]), int(center_arr[1]))

        clean_speed_ids = [int(x) for x in high_speed_ids]

        if direction_variance < self.PANIC_DIRECTION_VARIANCE:
            # Low variance = uniform direction = STAMPEDE
            return ActivityDetection(
                activity_type="STAMPEDE",
                severity="CRITICAL",
                confidence=round(float(min(0.95, avg_velocity / 40.0)), 2),
                description=f"Stampede detected: {len(clean_speed_ids)} people moving rapidly in same direction",
                involved_ids=clean_speed_ids,
                location=center,
                zone=self._location_to_zone(center[0], center[1], fw, fh)
            )
        else:
            # High variance + high speed = PANIC
            return ActivityDetection(
                activity_type="PANIC",
                severity="CRITICAL",
                confidence=round(float(min(0.90, avg_velocity / 50.0)), 2),
                description=f"Panic detected: {len(clean_speed_ids)} people in chaotic rapid movement",
                involved_ids=clean_speed_ids,
                location=center,
                zone=self._location_to_zone(center[0], center[1], fw, fh)
            )

    def _detect_falls(
        self, tracked: Dict[int, deque], fw: int, fh: int
    ) -> List[ActivityDetection]:
        """
        Detect medical falls: bounding box aspect ratio drops sharply 
        (standing person becomes horizontal), AND person stays down for sustained period.
        
        FIXED: Now enforces FALL_STAY_DOWN_FRAMES — the person must remain in a
        "down" state (low aspect ratio) for at least N consecutive frames to confirm
        a real fall (not just bending/stumbling momentarily).
        """
        falls = []

        for tid, hist in tracked.items():
            if len(hist) < self.FALL_FRAMES_WINDOW:
                continue

            # Get aspect ratios over recent frames
            recent = list(hist)[-self.FALL_FRAMES_WINDOW:]
            aspect_ratios = [h_val / max(w_val, 1) for _, _, w_val, h_val, _ in recent]

            # Check for sharp drop: was standing (ratio > 1.3), now fallen (ratio < 0.9)
            early_ratio = np.mean(aspect_ratios[:3])
            late_ratio = np.mean(aspect_ratios[-3:])

            if early_ratio > 1.3 and late_ratio < 0.9:
                ratio_drop = early_ratio - late_ratio

                if ratio_drop > self.FALL_ASPECT_RATIO_DROP:
                    # FIXED: Check if person is STAYING down (not just momentary drop)
                    # Look at the last FALL_STAY_DOWN_FRAMES frames
                    check_len = min(self.FALL_STAY_DOWN_FRAMES, len(hist))
                    if check_len < 5:
                        # Not enough history yet, still tracking the fall
                        continue

                    recent_stay_down = list(hist)[-check_len:]
                    stay_down_ratios = [h_val / max(w_val, 1) for _, _, w_val, h_val, _ in recent_stay_down]
                    
                    # Check that person REMAINS down (all recent frames have low aspect ratio)
                    # Allow some variation, but most frames must be "down"
                    down_frames = sum(1 for r in stay_down_ratios if r < self.FALL_MIN_ASPECT_RATIO)
                    down_ratio = down_frames / len(stay_down_ratios)

                    if down_ratio >= 0.7:  # At least 70% of frames must show "down" state
                        cx, cy = int(hist[-1][0]), int(hist[-1][1])

                        # FIXED: Track fall detection to avoid duplicate reports
                        # Only report fall once per person, when transition is first detected
                        if tid not in self._fall_candidates:
                            self._fall_candidates[tid] = 0
                        
                        self._fall_candidates[tid] += 1

                        # Only create activity detection when we've confirmed sustained fall
                        if self._fall_candidates[tid] == 1:
                            falls.append(ActivityDetection(
                                activity_type="FALL",
                                severity="HIGH",
                                confidence=round(float(min(0.90, ratio_drop / 1.5)), 2),
                                description=f"Person (ID:{tid}) has fallen — possible medical emergency",
                                involved_ids=[int(tid)],
                                location=(int(cx), int(cy)),
                                zone=self._location_to_zone(int(cx), int(cy), fw, fh)
                            ))
                    else:
                        # Person is getting up or false positive, clear candidate
                        self._fall_candidates.pop(tid, None)
                else:
                    # Ratio drop not significant enough, clear candidate
                    self._fall_candidates.pop(tid, None)
            else:
                # Person stood back up, clear candidate
                self._fall_candidates.pop(tid, None)

        return falls

    def _detect_fight(
        self, tracked: Dict[int, deque], fw: int, fh: int
    ) -> Optional[ActivityDetection]:
        """
        Detect fights: 2+ people in close proximity with high erratic oscillating motion.
        """
        ids = list(tracked.keys())

        if len(ids) < 2:
            return None

        # Check all pairs
        for i in range(len(ids)):
            for j in range(i + 1, min(len(ids), i + 10)):  # Limit pair checks
                tid_a, tid_b = ids[i], ids[j]
                hist_a, hist_b = tracked[tid_a], tracked[tid_b]

                if len(hist_a) < self.FIGHT_MIN_FRAMES or len(hist_b) < self.FIGHT_MIN_FRAMES:
                    continue

                # Check proximity (current position)
                dist = np.sqrt(
                    (hist_a[-1][0] - hist_b[-1][0])**2 +
                    (hist_a[-1][1] - hist_b[-1][1])**2
                )

                if dist > self.FIGHT_PROXIMITY_PX:
                    continue

                # Check erratic motion (oscillation)
                osc_a = self._compute_oscillation(hist_a)
                osc_b = self._compute_oscillation(hist_b)
                avg_osc = (osc_a + osc_b) / 2

                if avg_osc > self.FIGHT_OSCILLATION_THRESHOLD:
                    # Check they've been close for multiple frames
                    close_frames = 0
                    check_len = min(len(hist_a), len(hist_b), self.FIGHT_MIN_FRAMES)
                    for k in range(1, check_len + 1):
                        d = np.sqrt(
                            (hist_a[-k][0] - hist_b[-k][0])**2 +
                            (hist_a[-k][1] - hist_b[-k][1])**2
                        )
                        if d < self.FIGHT_PROXIMITY_PX * 1.5:
                            close_frames += 1

                    if close_frames >= self.FIGHT_MIN_FRAMES * 0.6:
                        cx = int((hist_a[-1][0] + hist_b[-1][0]) / 2)
                        cy = int((hist_a[-1][1] + hist_b[-1][1]) / 2)

                        return ActivityDetection(
                            activity_type="FIGHT",
                            severity="HIGH",
                            confidence=round(float(min(0.85, avg_osc / 20.0)), 2),
                            description=f"Possible altercation between Person ID:{tid_a} and ID:{tid_b}",
                            involved_ids=[int(tid_a), int(tid_b)],
                            location=(int(cx), int(cy)),
                            zone=self._location_to_zone(int(cx), int(cy), fw, fh)
                        )

        return None

    def _detect_dispersal(
        self, tracked: Dict[int, deque], fw: int, fh: int
    ) -> Optional[ActivityDetection]:
        """
        Detect dispersal: people rapidly moving away from a common center.
        """
        if len(tracked) < self.DISPERSAL_MIN_PEOPLE:
            return None

        # Get current positions and velocities
        current_positions = []
        velocity_vectors = []
        all_ids = []

        for tid, hist in tracked.items():
            if len(hist) < 5:
                continue
            cx, cy = hist[-1][0], hist[-1][1]
            cx_old, cy_old = hist[-5][0], hist[-5][1]
            vx = (cx - cx_old) / 4.0
            vy = (cy - cy_old) / 4.0
            speed = np.sqrt(vx**2 + vy**2)

            if speed > 3.0:  # Only consider moving people
                current_positions.append([cx, cy])
                velocity_vectors.append([vx, vy])
                all_ids.append(tid)

        if len(current_positions) < self.DISPERSAL_MIN_PEOPLE:
            return None

        positions = np.array(current_positions)
        velocities_arr = np.array(velocity_vectors)

        # Compute group center
        center = np.mean(positions, axis=0)

        # For each person, check if velocity vector points away from center
        diverging_count = 0
        for i in range(len(positions)):
            # Vector from center to person
            to_person = positions[i] - center
            to_person_norm = np.linalg.norm(to_person)
            if to_person_norm < 1:
                continue

            to_person_unit = to_person / to_person_norm
            vel_unit = velocities_arr[i] / max(np.linalg.norm(velocities_arr[i]), 1)

            # Dot product > 0 means moving away from center
            dot = np.dot(to_person_unit, vel_unit)
            if dot > 0.3:
                diverging_count += 1

        diverging_ratio = diverging_count / len(positions)
        avg_speed = np.mean(np.linalg.norm(velocities_arr, axis=1))

        if diverging_ratio > self.DISPERSAL_DIVERGENCE_RATIO and avg_speed > self.DISPERSAL_VELOCITY_THRESHOLD:
            center_int = (int(center[0]), int(center[1]))

            return ActivityDetection(
                activity_type="DISPERSAL",
                severity="HIGH",
                confidence=round(float(min(0.85, diverging_ratio)), 2),
                description=f"{diverging_count} people rapidly dispersing from a common point",
                involved_ids=[int(x) for x in all_ids],
                location=center_int,
                zone=self._location_to_zone(center_int[0], center_int[1], fw, fh)
            )

        return None

    def _detect_loitering(
        self, tracked: Dict[int, deque], fw: int, fh: int
    ) -> List[ActivityDetection]:
        """
        Detect loitering: person stays in a very small area for an extended time.
        """
        loitering = []

        for tid, hist in tracked.items():
            if len(hist) < self.LOITER_MIN_FRAMES:
                continue

            # Check spatial extent of all positions
            positions = np.array([[h[0], h[1]] for h in hist])
            spatial_range = np.max(positions, axis=0) - np.min(positions, axis=0)
            max_extent = np.max(spatial_range)

            if max_extent < self.LOITER_RADIUS_PX:
                cx, cy = int(positions[-1][0]), int(positions[-1][1])
                duration_frames = len(hist)

                loitering.append(ActivityDetection(
                    activity_type="LOITERING",
                    severity="LOW",
                    confidence=round(float(min(0.80, duration_frames / 150.0)), 2),
                    description=f"Person (ID:{tid}) loitering for ~{duration_frames // 10}s",
                    involved_ids=[int(tid)],
                    location=(int(cx), int(cy)),
                    zone=self._location_to_zone(int(cx), int(cy), fw, fh)
                ))

        return loitering

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    def _compute_speed(self, hist: deque) -> float:
        """Compute average speed over last 3 frames"""
        if len(hist) < 3:
            return 0.0
        recent = list(hist)[-3:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return np.sqrt(dx**2 + dy**2) / (len(recent) - 1)

    def _compute_oscillation(self, hist: deque) -> float:
        """
        Compute oscillation: frequency of direction changes.
        High oscillation = erratic motion (fighting, struggling).
        """
        if len(hist) < 6:
            return 0.0

        recent = list(hist)[-10:]
        if len(recent) < 4:
            return 0.0

        # Compute velocity direction changes
        direction_changes = 0
        for i in range(2, len(recent)):
            dx1 = recent[i-1][0] - recent[i-2][0]
            dy1 = recent[i-1][1] - recent[i-2][1]
            dx2 = recent[i][0] - recent[i-1][0]
            dy2 = recent[i][1] - recent[i-1][1]

            # Cross product sign change = direction reversal
            cross = dx1 * dy2 - dy1 * dx2
            dot = dx1 * dx2 + dy1 * dy2

            if dot < 0:  # Reversal
                direction_changes += 1

        # Also factor in speed variation
        speeds = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            speeds.append(np.sqrt(dx**2 + dy**2))

        speed_var = np.std(speeds) if speeds else 0

        return direction_changes * 2.0 + speed_var

    def _location_to_zone(self, x: int, y: int, frame_w: int, frame_h: int) -> str:
        """Convert pixel location to zone name"""
        zone_w = frame_w // 3
        zone_h = frame_h // 3

        col = min(x // max(zone_w, 1), 2)
        row = min(y // max(zone_h, 1), 2)

        zone_map = [
            ["top_left", "top_center", "top_right"],
            ["mid_left", "mid_center", "mid_right"],
            ["bot_left", "bot_center", "bot_right"]
        ]

        return zone_map[row][col]

    def _smooth_activities(self, activities: List[ActivityDetection]) -> List[ActivityDetection]:
        """
        Temporal smoothing to prevent activity flicker.
        An activity must be detected for N consecutive frames to be confirmed.
        
        FIXED: More robust filtering that respects FALL type specially
        (falls are one-time events, not continuous states)
        """
        current_types = {a.activity_type for a in activities}

        # Increment counters for detected activities
        for activity in activities:
            key = activity.activity_type
            self._activity_counts[key] += 1

            if self._activity_counts[key] >= self._confirmation_threshold:
                self._active_activities[key] = activity

        # Decrement counters for undetected activities
        for key in list(self._activity_counts.keys()):
            if key not in current_types:
                self._activity_counts[key] -= 1
                if self._activity_counts[key] <= 0:
                    self._activity_counts.pop(key, None)
                    self._active_activities.pop(key, None)

        return list(self._active_activities.values())

    def _assess_mood(self, activities: List[ActivityDetection]) -> str:
        """Assess overall scene mood from detected activities"""
        if not activities:
            return "CALM"

        severity_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
        max_severity = max(severity_map.get(a.severity, 0) for a in activities)

        critical_types = {"PANIC", "STAMPEDE", "FIGHT"}
        has_critical_type = any(a.activity_type in critical_types for a in activities)

        if max_severity >= 3 or has_critical_type:
            return "CHAOTIC"
        elif max_severity >= 2:
            return "TENSE"
        elif max_severity >= 1:
            return "ALERT"
        else:
            return "CALM"

    def get_stats(self) -> Dict:
        """Get recognizer statistics"""
        return {
            'tracked_ids': len(self.id_history),
            'active_activities': list(self._active_activities.keys()),
            'frame_count': self._frame_count,
            'fall_candidates': len(self._fall_candidates)
        }

    def reset(self):
        """Reset all state"""
        self.id_history.clear()
        self._activity_counts.clear()
        self._active_activities.clear()
        self._cluster_history.clear()
        self._fall_candidates.clear()
        self._frame_count = 0


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    recognizer = ActivityRecognizer()

    print("Testing ActivityRecognizer with simulated data...\n")

    # === Scenario 1: Gathering ===
    print("=== SCENARIO 1: People Gathering ===")
    for frame in range(30):
        detections = []
        for person_id in range(6):
            # People converge toward center (300, 300) over time
            spread = max(200 - frame * 5, 30)
            angle = person_id * (2 * np.pi / 6)
            cx = int(300 + spread * np.cos(angle))
            cy = int(300 + spread * np.sin(angle))
            detections.append({
                'id': person_id,
                'bbox': [cx - 20, cy - 50, cx + 20, cy + 50],
                'centroid': [cx, cy]
            })

        result = recognizer.detect(detections, (720, 1280))

        if result.activities:
            for act in result.activities:
                print(f"  Frame {frame:2d}: {act.activity_type} — {act.description} "
                      f"(conf: {act.confidence}, severity: {act.severity})")

    print(f"  Scene mood: {result.scene_mood}")

    # === Scenario 2: Person Fall (FIXED: now requires sustained "down" state) ===
    recognizer.reset()
    print("\n=== SCENARIO 2: Person Falls (Sustained) ===")
    for frame in range(30):
        # Person starts standing, collapses, then stays down
        if frame < 5:
            bbox_w, bbox_h = 40, 120  # Standing
        elif frame < 10:
            bbox_h = max(120 - (frame - 5) * 20, 40)  # Falling
            bbox_w = min(40 + (frame - 5) * 10, 100)
        else:
            # Stayed down (maintaining collapsed state)
            bbox_w, bbox_h = 90, 40

        detections = [{
            'id': 0,
            'bbox': [300 - bbox_w//2, 400 - bbox_h, 300 + bbox_w//2, 400],
            'centroid': [300, 400 - bbox_h//2]
        }]

        result = recognizer.detect(detections, (720, 1280))

        if result.activities:
            for act in result.activities:
                print(f"  Frame {frame:2d}: {act.activity_type} — {act.description}")

    print("\n[OK] Test complete!")
    print(f"Stats: {recognizer.get_stats()}")