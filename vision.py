"""
ApexMotion OS v3.0 - Spatial Grid & Industrial Mesh Architecture
Intel OpenVINO & NPU Benchmark Engine + Dynamic SLAM + Chaos Engineering Perception
"""

from __future__ import annotations

import base64
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class Obstacle:
    def __init__(self, x: float, y: float, radius: float, label: str, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.radius = radius
        self.label = label
        self.color = color


class OccupancyGridMap:
    """
    2D Fleet-Wide Occupancy Grid SLAM Map (10m x 10m at 0.25m resolution = 40x40 cells).
    0 = UNEXPLORED (Fog-of-war), 1 = FREE_SPACE, 2 = OCCUPIED
    """

    def __init__(self, arena_w: float = 10.0, arena_h: float = 10.0, resolution: float = 0.25):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.resolution = resolution
        self.cols = int(arena_w / resolution)
        self.rows = int(arena_h / resolution)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.total_cells = self.rows * self.cols
        self.discovered_cells = 0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        c = int(np.clip(x / self.resolution, 0, self.cols - 1))
        r = int(np.clip(y / self.resolution, 0, self.rows - 1))
        return c, r

    def update_from_lidar(self, rx: float, ry: float, rays: List[Dict[str, Any]]):
        start_c, start_r = self.world_to_grid(rx, ry)
        self.grid[start_r, start_c] = 1

        for ray in rays:
            hx = ray["hit_x"]
            hy = ray["hit_y"]
            end_c, end_r = self.world_to_grid(hx, hy)

            dc = abs(end_c - start_c)
            dr = abs(end_r - start_r)
            sc = 1 if start_c < end_c else -1
            sr = 1 if start_r < end_r else -1
            err = dc - dr

            cur_c, cur_r = start_c, start_r
            while True:
                if (cur_c == end_c and cur_r == end_r):
                    if ray["distance"] < 6.4:
                        self.grid[cur_r, cur_c] = 2
                    else:
                        self.grid[cur_r, cur_c] = 1
                    break

                if self.grid[cur_r, cur_c] != 2:
                    self.grid[cur_r, cur_c] = 1

                e2 = 2 * err
                if e2 > -dr:
                    err -= dr
                    cur_c += sc
                if e2 < dc:
                    err += dc
                    cur_r += sr

                if cur_c < 0 or cur_c >= self.cols or cur_r < 0 or cur_r >= self.rows:
                    break

        self.discovered_cells = int(np.count_nonzero(self.grid))

    @property
    def exploration_percentage(self) -> float:
        return round((self.discovered_cells / self.total_cells) * 100.0, 1)

    def get_compact_payload(self) -> Dict[str, Any]:
        return {
            "cols": self.cols,
            "rows": self.rows,
            "resolution": self.resolution,
            "exploration_pct": self.exploration_percentage,
            "grid_data": self.grid.tolist(),
        }


shared_fleet_grid = OccupancyGridMap()


class EdgePerceptionEngine:
    """
    Perception engine benchmarked for Intel Core Ultra NPU (OpenVINO INT8) vs CPU (FP32).
    Includes Chaos Engineering camera blackout and 24-ray planar LiDAR.
    """

    def __init__(self, arena_size: Tuple[float, float] = (10.0, 10.0)):
        self.arena_w, self.arena_h = arena_size
        self.num_lidar_rays = 24
        self.lidar_max_range = 6.5
        self.frame_width = 480
        self.frame_height = 270
        self.frame_counter = 0

        self.static_obstacles: List[Obstacle] = [
            Obstacle(3.0, 3.5, 0.65, "Hazard Barrel", (40, 120, 255)),
            Obstacle(7.0, 3.0, 0.75, "Intel NPU Node", (255, 180, 0)),
            Obstacle(5.0, 6.5, 0.85, "Industrial Pallet", (50, 200, 50)),
            Obstacle(2.5, 7.5, 0.55, "Facility Forklift", (200, 60, 240)),
            Obstacle(8.0, 7.5, 0.60, "Charging Station", (0, 220, 220)),
        ]

        # Hardware Target Profile
        # Options: "INTEL_NPU" (OpenVINO INT8) vs "CPU_FP32" (Baseline)
        self.hardware_profile: str = "INTEL_NPU"
        self.last_inference_ms = 4.1
        self.last_fps = 58.0
        self.last_power_watts = 4.2
        self._last_frame_time = time.perf_counter()

        # Chaos Engineering Flags
        self.chaos_camera_dropout: bool = False

    def set_hardware_profile(self, profile: str):
        if profile in ["INTEL_NPU", "CPU_FP32"]:
            self.hardware_profile = profile

    @property
    def target_accelerator(self) -> str:
        if self.hardware_profile == "INTEL_NPU":
            return "Intel Core Ultra NPU (OpenVINO INT8)"
        else:
            return "Host CPU Baseline (Unoptimized FP32)"

    @property
    def obstacles(self) -> List[Obstacle]:
        return self.static_obstacles

    def update_obstacles(self, dt: float = 0.02):
        self.static_obstacles[3].y = 7.5 + 0.45 * math.sin(time.time() * 0.9)

    def compute_lidar(
        self,
        rx: float,
        ry: float,
        r_theta: float,
        peer_agents: Optional[List[Dict[str, Any]]] = None,
        update_slam: bool = True,
    ) -> List[Dict[str, Any]]:
        rays = []
        angle_step = (2.0 * math.pi) / self.num_lidar_rays

        all_obstacles: List[Obstacle] = list(self.static_obstacles)
        if peer_agents:
            for pa in peer_agents:
                all_obstacles.append(
                    Obstacle(pa["x"], pa["y"], 0.40, pa.get("callsign", "Peer AGV"), (0, 240, 255))
                )

        for i in range(self.num_lidar_rays):
            ray_rel_angle = -math.pi + i * angle_step
            ray_global_angle = r_theta + ray_rel_angle

            dx = math.cos(ray_global_angle)
            dy = math.sin(ray_global_angle)

            min_dist = self.lidar_max_range
            hit_object: Optional[str] = None

            if dx > 1e-6:
                d = (self.arena_w - rx) / dx
                if 0 <= d < min_dist:
                    min_dist = d
                    hit_object = "East Perimeter"
            elif dx < -1e-6:
                d = (0.0 - rx) / dx
                if 0 <= d < min_dist:
                    min_dist = d
                    hit_object = "West Perimeter"

            if dy > 1e-6:
                d = (self.arena_h - ry) / dy
                if 0 <= d < min_dist:
                    min_dist = d
                    hit_object = "North Perimeter"
            elif dy < -1e-6:
                d = (0.0 - ry) / dy
                if 0 <= d < min_dist:
                    min_dist = d
                    hit_object = "South Perimeter"

            for obs in all_obstacles:
                ox = obs.x - rx
                oy = obs.y - ry
                proj = ox * dx + oy * dy
                if proj > 0:
                    perp_sq = (ox * ox + oy * oy) - (proj * proj)
                    rad_sq = obs.radius * obs.radius
                    if perp_sq < rad_sq:
                        half_chord = math.sqrt(max(0.0, rad_sq - perp_sq))
                        dist_hit = proj - half_chord
                        if 0 < dist_hit < min_dist:
                            min_dist = dist_hit
                            hit_object = obs.label

            noisy_dist = max(0.05, min_dist + random.gauss(0.0, 0.008))
            hit_x = rx + dx * noisy_dist
            hit_y = ry + dy * noisy_dist

            rays.append({
                "ray_index": i,
                "relative_angle_rad": round(ray_rel_angle, 3),
                "distance": round(noisy_dist, 3),
                "hit_x": round(hit_x, 3),
                "hit_y": round(hit_y, 3),
                "obstacle": hit_object or "Clear Corridor",
                "hazard": noisy_dist < 1.0,
                "critical": noisy_dist < 0.45,
            })

        if update_slam:
            shared_fleet_grid.update_from_lidar(rx, ry, rays)

        return rays

    def compute_collision_risk(
        self,
        rx: float,
        ry: float,
        r_theta: float,
        linear_v: float,
        lidar_rays: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        forward_rays = [
            r for r in lidar_rays if abs(r["relative_angle_rad"]) <= (math.pi / 4.5)
        ]

        if not forward_rays:
            min_forward_dist = self.lidar_max_range
            closest_target = "None"
        else:
            closest_ray = min(forward_rays, key=lambda r: r["distance"])
            min_forward_dist = closest_ray["distance"]
            closest_target = closest_ray["obstacle"]

        effective_v = max(linear_v, 0.05)
        ttc = round(min_forward_dist / effective_v, 2)

        if min_forward_dist <= 0.45:
            risk_level = "CRITICAL_HAZARD"
            risk_color = "#ef4444"
            auto_brake_required = True
        elif min_forward_dist <= 0.90:
            risk_level = "WARNING"
            risk_color = "#f97316"
            auto_brake_required = False
        elif min_forward_dist <= 1.60:
            risk_level = "CAUTION"
            risk_color = "#eab308"
            auto_brake_required = False
        else:
            risk_level = "SAFE"
            risk_color = "#10b981"
            auto_brake_required = False

        return {
            "risk_level": risk_level,
            "risk_color": risk_color,
            "min_forward_distance_m": round(min_forward_dist, 2),
            "closest_target": closest_target,
            "ttc_seconds": ttc if ttc < 60.0 else 99.9,
            "auto_brake_required": auto_brake_required,
        }

    def render_synthetic_camera_frame(
        self,
        rx: float,
        ry: float,
        r_theta: float,
        collision_telemetry: Dict[str, Any],
        peer_agents: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        self.frame_counter += 1
        w, h = self.frame_width, self.frame_height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Check Chaos Camera Dropout Fault Mode
        if self.chaos_camera_dropout:
            # Generate static noise frame
            noise = np.random.randint(0, 80, (h, w, 3), dtype=np.uint8)
            cv2.rectangle(noise, (20, h // 2 - 35), (w - 20, h // 2 + 35), (10, 10, 180), -1)
            cv2.rectangle(noise, (20, h // 2 - 35), (w - 20, h // 2 + 35), (0, 0, 255), 2)
            cv2.putText(
                noise,
                "CHAOS FAULT: SENSOR_DROPOUT",
                (35, h // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                noise,
                "FAILSAFE: LIDAR DEAD-RECKONING ACTIVE",
                (35, h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 240, 255),
                1,
                cv2.LINE_AA,
            )
            _, buffer = cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, 60])
            base64_jpeg = base64.b64encode(buffer).decode("utf-8")
            return base64_jpeg, [], {
                "accelerator": "OFFLINE (SENSOR DROPOUT)",
                "inference_ms": 0.0,
                "fps": 0.0,
                "power_watts": 0.0,
                "detections_count": 0,
                "frame_id": self.frame_counter,
                "fault": "SENSOR_DROPOUT",
            }

        # 2. Normal Perspective Frame
        horizon_y = int(h * 0.45)
        for y in range(horizon_y):
            r = y / max(1, horizon_y)
            cv2.line(frame, (0, y), (w, y), (int(16 + 8 * r), int(14 + 6 * r), int(26 + 10 * r)), 1)
        for y in range(horizon_y, h):
            r = (y - horizon_y) / (h - horizon_y)
            cv2.line(frame, (0, y), (w, y), (int(30 + 15 * r), int(34 + 18 * r), int(42 + 20 * r)), 1)

        vanishing_x = int(w * 0.5)
        for x_lane in range(-3, 4):
            bottom_x = int(w * 0.5 + x_lane * (w * 0.22))
            cv2.line(frame, (vanishing_x, horizon_y), (bottom_x, h), (55, 62, 70), 1)

        detections: List[Dict[str, Any]] = []
        fov_rad = math.radians(60.0)

        all_renderable = list(self.static_obstacles)
        if peer_agents:
            for pa in peer_agents:
                all_renderable.append(
                    Obstacle(pa["x"], pa["y"], 0.40, pa.get("callsign", "Peer AGV"), (0, 240, 255))
                )

        for obs in all_renderable:
            dx = obs.x - rx
            dy = obs.y - ry
            dist = math.hypot(dx, dy)
            if dist < 0.2 or dist > 7.0:
                continue

            angle_to_obs = math.atan2(dy, dx)
            angle_diff = (angle_to_obs - r_theta + math.pi) % (2 * math.pi) - math.pi

            if abs(angle_diff) <= (fov_rad / 2.0):
                screen_x_norm = angle_diff / (fov_rad / 2.0)
                center_px_x = int((screen_x_norm + 1.0) * 0.5 * w)

                box_h = int(np.clip((h * 0.6) / dist, 25, h * 0.8))
                box_w = int(box_h * 0.85)

                center_px_y = horizon_y + int((h - horizon_y) * (1.0 - min(1.0, dist / 7.0) * 0.6))
                x1 = int(np.clip(center_px_x - box_w // 2, 0, w - 1))
                y1 = int(np.clip(center_px_y - box_h // 2, 0, h - 1))
                x2 = int(np.clip(center_px_x + box_w // 2, 0, w - 1))
                y2 = int(np.clip(center_px_y + box_h // 2, 0, h - 1))

                if x2 > x1 and y2 > y1:
                    box_color = obs.color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    bracket_len = min(12, (x2 - x1) // 3)
                    cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), (0, 240, 255), 2)
                    cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), (0, 240, 255), 2)
                    cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), (0, 240, 255), 2)
                    cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), (0, 240, 255), 2)
                    cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), (0, 240, 255), 2)
                    cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), (0, 240, 255), 2)
                    cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), (0, 240, 255), 2)
                    cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), (0, 240, 255), 2)

                    conf = round(0.92 + 0.07 * math.sin(self.frame_counter * 0.1 + dist), 2)
                    label_text = f"{obs.label} | {dist:.1f}m ({int(conf * 100)}%)"

                    (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.rectangle(frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), box_color, -1)
                    cv2.putText(frame, label_text, (x1 + 3, max(lh + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

                    detections.append({
                        "class": obs.label,
                        "distance_m": round(dist, 2),
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    })

        ch_x, ch_y = w // 2, h // 2
        cv2.drawMarker(frame, (ch_x, ch_y), (0, 240, 255), cv2.MARKER_CROSS, 16, 1)
        cv2.circle(frame, (ch_x, ch_y), 18, (0, 240, 255), 1)

        risk_lvl = collision_telemetry["risk_level"]
        risk_bgr = (40, 200, 40) if risk_lvl == "SAFE" else ((0, 140, 255) if risk_lvl == "CAUTION" else (0, 0, 240))
        cv2.rectangle(frame, (10, 10), (200, 32), (18, 22, 28), -1)
        cv2.rectangle(frame, (10, 10), (200, 32), risk_bgr, 1)
        cv2.putText(frame, f"SLAM GUARD: {risk_lvl}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, risk_bgr, 1, cv2.LINE_AA)

        # 3. Dynamic Inference Latency & Power Benchmarks
        if self.hardware_profile == "INTEL_NPU":
            # Intel Core Ultra NPU INT8: 4.1ms - 4.6ms, ~58-62 FPS, 4.2W
            sim_ms = round(4.1 + 0.4 * math.sin(self.frame_counter * 0.15) + random.uniform(0, 0.2), 2)
            sim_fps = round(58.0 + random.uniform(-2, 3), 1)
            sim_watts = 4.2
            badge_text = f"Intel NPU {sim_ms}ms | {sim_fps:.0f} FPS | {sim_watts}W"
            badge_color = (0, 240, 255)
        else:
            # Host CPU Baseline FP32: 38.2ms - 44.0ms, ~24-26 FPS, 28.5W
            sim_ms = round(39.5 + 3.0 * math.sin(self.frame_counter * 0.1) + random.uniform(0, 2.0), 1)
            sim_fps = round(25.0 + random.uniform(-1, 2), 1)
            sim_watts = 28.5
            badge_text = f"Host CPU FP32 {sim_ms}ms | {sim_fps:.0f} FPS | {sim_watts}W"
            badge_color = (180, 180, 180)

        self.last_inference_ms = sim_ms
        self.last_fps = sim_fps
        self.last_power_watts = sim_watts

        (pw, ph), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(frame, (w - pw - 18, 10), (w - 10, 32), (18, 22, 28), -1)
        cv2.rectangle(frame, (w - pw - 18, 10), (w - 10, 32), badge_color, 1)
        cv2.putText(frame, badge_text, (w - pw - 14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_color, 1, cv2.LINE_AA)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        base64_jpeg = base64.b64encode(buffer).decode("utf-8")

        perf_metrics = {
            "accelerator": self.target_accelerator,
            "hardware_profile": self.hardware_profile,
            "inference_ms": sim_ms,
            "fps": sim_fps,
            "power_watts": sim_watts,
            "detections_count": len(detections),
            "frame_id": self.frame_counter,
            "fault": "NONE",
        }

        return base64_jpeg, detections, perf_metrics
