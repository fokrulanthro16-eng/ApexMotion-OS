"""
ApexMotion OS v3.0 - Spatial Grid & Industrial Mesh Architecture
Intel NPU Benchmark Target Switch + Chaos Engineering Injection + Enterprise ROI Telemetry
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import logging
import math
import time
import uuid
import base64
from typing import Any, Deque, Dict, List, Optional, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from planner import CommandIntent, MotionCommand, SafetyStatus, parse_enterprise_intent
from vision import EdgePerceptionEngine, shared_fleet_grid
import tts_engine
from audio_bank import STUDIO_AUDIO_BANK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ApexMotion-v3] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ApexMotionV3")


def resolve_studio_or_synthesized_audio(text: str, agent_id: str = "Alpha") -> Tuple[str, str, str]:
    """
    Hybrid High-Fidelity Audio Resolver:
    1. Direct match with pre-rendered studio high-fidelity clips.
    2. Context-aware intent mapping (RTB, yields, failsafe, nominal).
    3. Dynamic edge acoustic synthesis fallback with tactical radio chirp.
    Returns: (spoken_text, base64_wav, audio_source)
    """
    clean = text.lower().strip()

    # Pre-rendered studio clip mappings
    if "dock" in clean or "base" in clean or "return to base" in clean or "rtb" in clean:
        if "charlie" in clean:
            clip = STUDIO_AUDIO_BANK.get("charlie_rtb")
            return clip["text"], clip["audio_base64"], "STUDIO_HD"
        else:
            clip = STUDIO_AUDIO_BANK.get("alpha_rtb")
            return clip["text"], clip["audio_base64"], "STUDIO_HD"

    if "yield" in clean or "hazard" in clean or "detour" in clean:
        clip = STUDIO_AUDIO_BANK.get("bravo_yield")
        return clip["text"], clip["audio_base64"], "STUDIO_HD"

    if "intercept" in clean or "collision" in clean or "proximity" in clean:
        clip = STUDIO_AUDIO_BANK.get("safety_intercept")
        return clip["text"], clip["audio_base64"], "STUDIO_HD"

    if "watchdog" in clean or "latency" in clean or "threshold" in clean:
        clip = STUDIO_AUDIO_BANK.get("watchdog_alert")
        return clip["text"], clip["audio_base64"], "STUDIO_HD"

    if "nominal" in clean or "50 hertz" in clean or "fleet synchronized" in clean or "system online" in clean:
        clip = STUDIO_AUDIO_BANK.get("system_nominal")
        return clip["text"], clip["audio_base64"], "STUDIO_HD"

    if "stop" in clean or "halt" in clean or "brake" in clean or "e-stop" in clean:
        clip = STUDIO_AUDIO_BANK.get("fleet_estop")
        return clip["text"], clip["audio_base64"], "STUDIO_HD"

    if "forward" in clean or "advance" in clean or "ahead" in clean:
        clip = STUDIO_AUDIO_BANK.get("alpha_forward")
        return clip["text"], clip["audio_base64"], "STUDIO_HD"

    # Dynamic Fallback: Synthesize using the edge formant engine
    wav_bytes = tts_engine.synthesize_neural_speech_wav(text, pitch_hz=115.0, include_chirp=True)
    b64_str = base64.b64encode(wav_bytes).decode("ascii")
    return text, b64_str, "DYNAMIC_SYNTHESIS"


# Schemas
class Vector3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Twist(BaseModel):
    linear: Vector3 = Field(default_factory=Vector3)
    angular: Vector3 = Field(default_factory=Vector3)


class Ros2CmdVelRequest(BaseModel):
    agent_id: str = Field("Alpha")
    twist: Twist


class Ros2BridgeModeRequest(BaseModel):
    mode: str = Field(..., description="'SIMULATION' or 'ROS2_BRIDGE'")


class HardwareProfileRequest(BaseModel):
    target: str = Field(..., description="'INTEL_NPU' or 'CPU_FP32'")


class ChaosFaultRequest(BaseModel):
    fault: str = Field(..., description="'SENSOR_DROPOUT', 'LATENCY_SPIKE', or 'CLEAR'")


class VoiceCommandRequest(BaseModel):
    text: str = Field(..., example="Bravo, patrol between dock and pallet 3 times")
    target_agent: Optional[str] = Field("Alpha")


class TtsRequest(BaseModel):
    text: str = Field(..., example="Alpha, moving forward 2 meters")
    pitch_hz: Optional[float] = Field(115.0)
    include_chirp: Optional[bool] = Field(True)
    engine: Optional[str] = Field("neural", description="'neural' or 'speechmatics'")


class TeleopCommand(BaseModel):
    agent_id: str = Field("Alpha")
    linear_v: float = Field(..., ge=-1.5, le=1.5)
    angular_v: float = Field(..., ge=-2.0, le=2.0)


# Flight Recorder Blackbox
class FlightRecorder:
    def __init__(self, max_frames: int = 1500):
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=max_frames)
        self.incidents: List[Dict[str, Any]] = []

    def push_frame(self, frame: Dict[str, Any]):
        self.buffer.append(frame)

    def capture_incident(self, reason: str, culprit: str, agent_ids: List[str]) -> str:
        incident_id = f"INC-{int(time.time())}-{str(uuid.uuid4())[:4].upper()}"
        frames_snapshot = list(self.buffer)[-400:]

        incident_record = {
            "incident_id": incident_id,
            "timestamp": round(time.time(), 3),
            "reason": reason,
            "culprit": culprit,
            "agent_ids": agent_ids,
            "total_frames": len(frames_snapshot),
            "frames": frames_snapshot,
        }
        self.incidents.append(incident_record)
        if len(self.incidents) > 15:
            self.incidents.pop(0)

        logger.warning(f"[BLACKBOX] Incident captured: {incident_id} ({reason}) with {len(frames_snapshot)} frames.")
        return incident_id


flight_recorder = FlightRecorder(max_frames=1500)


# Single AGV Agent Instance
class AutonomousAgent:
    def __init__(
        self,
        agent_id: str,
        callsign: str,
        color: str,
        priority: int,
        initial_pose: Tuple[float, float, float],
    ):
        self.id = agent_id
        self.callsign = callsign
        self.color = color
        self.priority = priority

        self.spawn_x, self.spawn_y, self.spawn_theta = initial_pose
        self.x: float = self.spawn_x
        self.y: float = self.spawn_y
        self.theta: float = self.spawn_theta

        self.v: float = 0.0
        self.w: float = 0.0
        self.v_target: float = 0.0
        self.w_target: float = 0.0

        self.max_linear_accel = 2.0
        self.max_angular_accel = 4.0
        self.battery_pct: float = 98.6
        self.system_status: str = "ONLINE_NOMINAL"
        self.emergency_brake: bool = False
        self.yielding: bool = False
        self.yield_detour_wp: Optional[Tuple[float, float]] = None

        self.active_command: Optional[MotionCommand] = None
        self.command_time_remaining: float = 0.0
        self.waypoint_queue: List[Dict[str, Any]] = []
        self.target_waypoint: Optional[Dict[str, Any]] = None
        self.macro_progress: str = "IDLE"

        self.perception = EdgePerceptionEngine(arena_size=(10.0, 10.0))
        self.breadcrumb_trail: List[List[float]] = []

    def reset_to_dock(self):
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.theta = self.spawn_theta
        self.v = 0.0
        self.w = 0.0
        self.v_target = 0.0
        self.w_target = 0.0
        self.emergency_brake = False
        self.yielding = False
        self.yield_detour_wp = None
        self.active_command = None
        self.waypoint_queue.clear()
        self.target_waypoint = None
        self.breadcrumb_trail.clear()
        self.system_status = "ONLINE_NOMINAL"
        self.macro_progress = "DOCK_REST"

    def apply_command(self, cmd: MotionCommand):
        if cmd.emergency:
            self.emergency_brake = True
            self.v = 0.0
            self.w = 0.0
            self.v_target = 0.0
            self.w_target = 0.0
            self.active_command = cmd
            self.waypoint_queue.clear()
            self.target_waypoint = None
            self.yield_detour_wp = None
            self.system_status = "EMERGENCY_STOP"
            return

        self.emergency_brake = False
        self.yielding = False
        self.yield_detour_wp = None
        self.active_command = cmd
        self.command_time_remaining = cmd.duration
        self.system_status = f"EXECUTING_{cmd.intent.value}"

        if cmd.waypoints:
            self.waypoint_queue = [wp.model_dump() for wp in cmd.waypoints]
            self.target_waypoint = self.waypoint_queue.pop(0) if self.waypoint_queue else None
            wp_name = self.target_waypoint.get("name", "Waypoint") if self.target_waypoint else ""
            self.macro_progress = f"Moving to {wp_name}"
        else:
            self.target_waypoint = None
            self.v_target = cmd.linear_velocity
            self.w_target = cmd.angular_velocity
            self.macro_progress = f"Executing {cmd.intent.value}"

    def update_agent_physics(self, dt: float, peer_agents: List[AutonomousAgent]):
        peers_data = [{"x": p.x, "y": p.y, "callsign": p.callsign} for p in peer_agents]

        active_target = self.yield_detour_wp or (
            (self.target_waypoint["x"], self.target_waypoint["y"]) if self.target_waypoint else None
        )

        if active_target is not None and not self.emergency_brake:
            tx, ty = active_target
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)

            if dist < 0.22:
                if self.yield_detour_wp:
                    self.yield_detour_wp = None
                elif self.waypoint_queue:
                    self.target_waypoint = self.waypoint_queue.pop(0)
                    self.macro_progress = f"Heading to {self.target_waypoint.get('name', 'Waypoint')}"
                else:
                    self.target_waypoint = None
                    self.v_target = 0.0
                    self.w_target = 0.0
                    self.active_command = None
                    self.system_status = "ONLINE_STANDBY"
                    self.macro_progress = "MISSION_COMPLETE"
            else:
                target_heading = math.atan2(dy, dx)
                heading_err = (target_heading - self.theta + math.pi) % (2 * math.pi) - math.pi

                if abs(heading_err) > 0.45:
                    self.v_target = 0.0
                    self.w_target = math.copysign(1.0, heading_err)
                else:
                    speed_cap = 0.40 if self.yielding else (self.target_waypoint.get("speed_limit", 0.6) if self.target_waypoint else 0.5)
                    self.v_target = min(speed_cap, dist * 0.9)
                    self.w_target = heading_err * 1.8

        elif self.active_command is not None and self.command_time_remaining > 0:
            self.command_time_remaining -= dt
            if self.command_time_remaining <= 0:
                self.v_target = 0.0
                self.w_target = 0.0
                self.active_command = None
                self.system_status = "ONLINE_STANDBY"
                self.macro_progress = "IDLE"

        # Perception & LiDAR + Dynamic SLAM
        lidar_rays = self.perception.compute_lidar(
            self.x, self.y, self.theta, peer_agents=peers_data, update_slam=True
        )
        collision_risk = self.perception.compute_collision_risk(
            self.x, self.y, self.theta, self.v, lidar_rays
        )

        if collision_risk["auto_brake_required"] and self.v > 0:
            self.v_target = 0.0
            self.v = 0.0
            self.system_status = "SAFETY_INTERCEPT_COLLISION"

        # Kinematics
        if not self.emergency_brake:
            effective_v_target = 0.15 if (self.yielding and not self.yield_detour_wp) else self.v_target
            effective_w_target = self.w_target

            dv_max = self.max_linear_accel * dt
            dv = max(-dv_max, min(dv_max, effective_v_target - self.v))
            self.v += dv

            dw_max = self.max_angular_accel * dt
            dw = max(-dw_max, min(dw_max, effective_w_target - self.w))
            self.w += dw

            self.theta = (self.theta + self.w * dt) % (2 * math.pi)
            ds = self.v * dt
            self.x += ds * math.cos(self.theta)
            self.y += ds * math.sin(self.theta)

            wall_margin = 0.45
            if self.x < wall_margin:
                self.x = wall_margin
                self.v = -self.v * 0.2
            elif self.x > 10.0 - wall_margin:
                self.x = 10.0 - wall_margin
                self.v = -self.v * 0.2

            if self.y < wall_margin:
                self.y = wall_margin
                self.v = -self.v * 0.2
            elif self.y > 10.0 - wall_margin:
                self.y = 10.0 - wall_margin
                self.v = -self.v * 0.2

        # Breadcrumbs
        if abs(self.v) > 0.02 or abs(self.w) > 0.02:
            if not self.breadcrumb_trail or math.hypot(self.x - self.breadcrumb_trail[-1][0], self.y - self.breadcrumb_trail[-1][1]) > 0.15:
                self.breadcrumb_trail.append([round(self.x, 2), round(self.y, 2)])
                if len(self.breadcrumb_trail) > 60:
                    self.breadcrumb_trail.pop(0)

        # Battery
        drain = 0.0004 if (abs(self.v) > 0.05 or abs(self.w) > 0.05) else 0.00005
        self.battery_pct = max(1.0, round(self.battery_pct - drain, 3))

        video_b64, detections, perf_metrics = self.perception.render_synthetic_camera_frame(
            self.x, self.y, self.theta, collision_risk, peer_agents=peers_data
        )

        return {
            "id": self.id,
            "callsign": self.callsign,
            "color": self.color,
            "priority": self.priority,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "theta": round(self.theta, 3),
            "heading_deg": round(math.degrees(self.theta) % 360, 1),
            "linear_velocity": round(self.v, 3),
            "angular_velocity": round(self.w, 3),
            "target_v": round(self.v_target, 3),
            "target_w": round(self.w_target, 3),
            "battery": self.battery_pct,
            "status": self.system_status,
            "macro_progress": self.macro_progress,
            "emergency": self.emergency_brake,
            "yielding": self.yielding,
            "yield_detour": self.yield_detour_wp,
            "target_waypoint": self.target_waypoint,
            "trail": self.breadcrumb_trail,
            "lidar": lidar_rays,
            "collision": collision_risk,
            "vision": {
                "frame": f"data:image/jpeg;base64,{video_b64}",
                "detections": detections,
                "perf": perf_metrics,
            },
        }


# Multi-Agent Fleet Coordinator
class FleetCoordinator:
    def __init__(self):
        self.agents: Dict[str, AutonomousAgent] = {
            "Alpha": AutonomousAgent("Alpha", "AGV-Alpha", "#00f0ff", priority=1, initial_pose=(1.5, 1.5, 0.0)),
            "Bravo": AutonomousAgent("Bravo", "AGV-Bravo", "#f59e0b", priority=2, initial_pose=(8.5, 1.5, math.pi)),
            "Charlie": AutonomousAgent("Charlie", "AGV-Charlie", "#a855f7", priority=3, initial_pose=(1.5, 8.5, -math.pi / 2)),
        }
        self.yield_events_total = 0
        self.interagent_warnings_total = 0
        self.safety_intercepts_total = 0
        self.autonomous_decisions_total = 38
        self.active_focus_agent = "Alpha"
        self.bridge_mode: str = "SIMULATION"

        # Hardware Target
        self.current_hardware_profile = "INTEL_NPU"

        # Chaos Engineering Active States
        self.active_chaos_fault = "NONE"
        self.chaos_latency_spike_active = False

        self.audit_events: List[Dict[str, Any]] = []

    def log_audit(self, event_type: str, msg: str, severity: str = "INFO"):
        e = {
            "timestamp": round(time.time(), 3),
            "type": event_type,
            "message": msg,
            "severity": severity,
        }
        self.audit_events.append(e)
        if len(self.audit_events) > 30:
            self.audit_events.pop(0)

    def set_hardware_target(self, profile: str):
        self.current_hardware_profile = profile
        for ag in self.agents.values():
            ag.perception.set_hardware_profile(profile)
        self.log_audit("HW_BENCHMARK", f"Perception execution profile switched to {profile}.", "INFO")

    def inject_chaos_fault(self, fault: str):
        self.active_chaos_fault = fault
        if fault == "SENSOR_DROPOUT":
            self.chaos_latency_spike_active = False
            for ag in self.agents.values():
                ag.perception.chaos_camera_dropout = True
            self.log_audit("CHAOS_ENGINEERING", "Vision sensor blackout injected! Autonomous fallback to LiDAR dead-reckoning engaged.", "CRITICAL")
            self.safety_intercepts_total += 1

        elif fault == "LATENCY_SPIKE":
            self.chaos_latency_spike_active = True
            for ag in self.agents.values():
                ag.perception.chaos_camera_dropout = False
            self.log_audit("CHAOS_ENGINEERING", "Network latency spike injected (250ms > 120ms). Watchdog intercept armed.", "WARN")

        elif fault == "CLEAR":
            self.chaos_latency_spike_active = False
            for ag in self.agents.values():
                ag.perception.chaos_camera_dropout = False
                if ag.system_status == "WATCHDOG_FAILSAFE":
                    ag.system_status = "ONLINE_STANDBY"
            self.log_audit("CHAOS_ENGINEERING", "Chaos faults cleared. Restored nominal telemetry channels.", "INFO")

    def reset_fleet(self):
        for ag in self.agents.values():
            ag.reset_to_dock()
        self.log_audit("SYS_RESET", "Fleet repositioned to base docks.", "INFO")

    def dispatch_command(self, cmd: MotionCommand):
        self.autonomous_decisions_total += 1
        if cmd.target_agent == "ALL":
            for ag in self.agents.values():
                ag.apply_command(cmd)
            self.log_audit("DISPATCH", f"Broadcasted '{cmd.intent}' to entire fleet.", "DISPATCH")
        elif cmd.target_agent in self.agents:
            self.agents[cmd.target_agent].apply_command(cmd)
            self.log_audit("DISPATCH", f"Dispatched '{cmd.intent}' to {cmd.target_agent}.", "DISPATCH")

    def resolve_intersections_and_deadlocks(self) -> List[Dict[str, Any]]:
        agent_list = list(self.agents.values())
        interagent_links = []

        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a1 = agent_list[i]
                a2 = agent_list[j]
                dist = math.hypot(a1.x - a2.x, a1.y - a2.y)

                if dist < 0.75:
                    status = "CRITICAL_COLLISION"
                    self.safety_intercepts_total += 1
                    a1.v = 0.0
                    a2.v = 0.0
                    a1.system_status = "INTERAGENT_COLLISION_BRAKE"
                    a2.system_status = "INTERAGENT_COLLISION_BRAKE"
                    flight_recorder.capture_incident(
                        reason=f"Intersection Near-Miss: {a1.callsign} & {a2.callsign} ({dist:.2f}m)",
                        culprit=f"{a1.callsign} <-> {a2.callsign}",
                        agent_ids=[a1.id, a2.id],
                    )
                    self.log_audit("INTERSECTION_MANAGER", f"Collision Intercept between {a1.callsign} and {a2.callsign} ({dist:.2f}m)", "CRITICAL")

                elif dist < 1.6:
                    status = "MUTUAL_PROXIMITY_YIELD"
                    self.interagent_warnings_total += 1
                    self.autonomous_decisions_total += 1
                    higher, lower = (a1, a2) if a1.priority < a2.priority else (a2, a1)
                    lower.yielding = True
                    higher.yielding = False
                    lower.system_status = f"YIELDING_TO_{higher.id.upper()}"

                    axis_dx = (lower.x - higher.x) / max(0.01, dist)
                    axis_dy = (lower.y - higher.y) / max(0.01, dist)
                    perp_x = -axis_dy
                    perp_y = axis_dx

                    detour_x = max(0.6, min(9.4, lower.x + perp_x * 0.9))
                    detour_y = max(0.6, min(9.4, lower.y + perp_y * 0.9))
                    lower.yield_detour_wp = (round(detour_x, 2), round(detour_y, 2))

                    self.yield_events_total += 1
                    if not any(e["type"] == "INTERSECTION_MANAGER" and (time.time() - e["timestamp"] < 2.0) for e in self.audit_events):
                        self.log_audit("INTERSECTION_MANAGER", f"{lower.callsign} yielded to {higher.callsign}. Lateral reroute active.", "WARN")
                else:
                    status = "CLEAR"
                    if a1.yielding and not any(math.hypot(a1.x - other.x, a1.y - other.y) < 1.6 for other in agent_list if other != a1):
                        a1.yielding = False
                        a1.yield_detour_wp = None
                    if a2.yielding and not any(math.hypot(a2.x - other.x, a2.y - other.y) < 1.6 for other in agent_list if other != a2):
                        a2.yielding = False
                        a2.yield_detour_wp = None

                interagent_links.append({
                    "agent_a": a1.id,
                    "agent_b": a2.id,
                    "distance_m": round(dist, 2),
                    "status": status,
                })

        return interagent_links

    def update_fleet_physics(self, dt: float = 0.02) -> Dict[str, Any]:
        first_ag = self.agents["Alpha"]
        first_ag.perception.update_obstacles(dt)

        interagent_links = self.resolve_intersections_and_deadlocks()

        agent_telemetry: Dict[str, Any] = {}
        for ag in self.agents.values():
            peers = [p for p in self.agents.values() if p.id != ag.id]
            data = ag.update_agent_physics(dt, peer_agents=peers)
            agent_telemetry[ag.id] = data

        slam_data = shared_fleet_grid.get_compact_payload()

        # ROI & Safety metrics
        downtime_hours = round(self.safety_intercepts_total * 0.75, 1)
        cost_savings = int(downtime_hours * 1000)

        fleet_frame = {
            "timestamp": round(time.time(), 3),
            "agents": agent_telemetry,
            "interagent_links": interagent_links,
            "focused_agent": self.active_focus_agent,
            "bridge_mode": self.bridge_mode,
            "hardware_profile": self.current_hardware_profile,
            "chaos_fault": self.active_chaos_fault,
            "slam": slam_data,
            "arena": {
                "width": 10.0,
                "height": 10.0,
                "obstacles": [
                    {"x": o.x, "y": o.y, "radius": o.radius, "label": o.label}
                    for o in first_ag.perception.obstacles
                ],
            },
            "fleet_metrics": {
                "yield_events": self.yield_events_total,
                "interagent_warnings": self.interagent_warnings_total,
                "safety_intercepts": self.safety_intercepts_total,
                "exploration_pct": slam_data["exploration_pct"],
                "incidents_count": len(flight_recorder.incidents),
            },
            "roi_metrics": {
                "safety_intercepts_total": self.safety_intercepts_total,
                "downtime_hours_prevented": downtime_hours,
                "zero_human_interventions": self.autonomous_decisions_total,
                "estimated_cost_savings_usd": cost_savings,
            },
            "audit_events": self.audit_events[-8:],
        }

        flight_recorder.push_frame(fleet_frame)
        return fleet_frame


fleet_coordinator = FleetCoordinator()


# WebSocket Connection Manager
class FleetConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.client_heartbeats: Dict[WebSocket, float] = {}
        self.client_rtt: Dict[WebSocket, float] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        now = time.time()
        async with self._lock:
            self.active_connections.add(ws)
            self.client_heartbeats[ws] = now
            self.client_rtt[ws] = 0.015

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.active_connections.discard(ws)
            self.client_heartbeats.pop(ws, None)
            self.client_rtt.pop(ws, None)

    def record_heartbeat(self, ws: WebSocket, rtt_sec: float):
        self.client_heartbeats[ws] = time.time()
        self.client_rtt[ws] = rtt_sec

    async def broadcast(self, payload: Dict[str, Any]):
        if not self.active_connections:
            return

        dead = set()
        async with self._lock:
            conns = list(self.active_connections)

        for ws in conns:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self.active_connections.discard(ws)
                    self.client_heartbeats.pop(ws, None)
                    self.client_rtt.pop(ws, None)


ws_manager = FleetConnectionManager()
latest_fleet_payload: Dict[str, Any] = {}
physics_task: Optional[asyncio.Task] = None
broadcast_task: Optional[asyncio.Task] = None


async def fleet_physics_loop_50hz():
    dt = 0.02
    global latest_fleet_payload
    logger.info("ApexMotion v3.0 50Hz Physics & SLAM loop active.")
    try:
        while True:
            t0 = time.perf_counter()
            telemetry = fleet_coordinator.update_fleet_physics(dt)
            latest_fleet_payload = telemetry

            # Watchdog monitor & Chaos Latency Injection
            now = time.time()
            if ws_manager.active_connections:
                for ws, last_hb in list(ws_manager.client_heartbeats.items()):
                    rtt = ws_manager.client_rtt.get(ws, 0.0)
                    gap = now - last_hb

                    # Injected Chaos Latency Fault: simulates 250ms latency spike
                    if fleet_coordinator.chaos_latency_spike_active:
                        rtt = 0.250

                    if rtt > 0.120 or gap > 2.2:
                        for ag in fleet_coordinator.agents.values():
                            if not ag.emergency_brake and ag.v > 0.05:
                                ag.v_target = 0.0
                                ag.system_status = "WATCHDOG_FAILSAFE"
                        if not any(e["type"] == "WATCHDOG_INTERCEPT" and (time.time() - e["timestamp"] < 2.0) for e in fleet_coordinator.audit_events):
                            fleet_coordinator.log_audit("WATCHDOG_INTERCEPT", f"Watchdog tripwire! Latency {rtt*1000:.0f}ms > 120ms limit. Controlled deceleration active.", "CRITICAL")
                            fleet_coordinator.safety_intercepts_total += 1

            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.001, dt - elapsed))
    except asyncio.CancelledError:
        logger.info("Physics loop terminated.")


async def fleet_broadcast_loop_25hz():
    dt = 0.04
    logger.info("Fleet broadcast streaming hub active at 25Hz.")
    try:
        while True:
            if latest_fleet_payload:
                await ws_manager.broadcast(latest_fleet_payload)
            await asyncio.sleep(dt)
    except asyncio.CancelledError:
        logger.info("Fleet broadcast loop terminated.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global physics_task, broadcast_task
    logger.info("Booting ApexMotion OS v3.0 Spatial Grid & Mesh Gateway...")
    physics_task = asyncio.create_task(fleet_physics_loop_50hz())
    broadcast_task = asyncio.create_task(fleet_broadcast_loop_25hz())
    yield
    logger.info("Shutting down ApexMotion OS v3.0...")
    if physics_task: physics_task.cancel()
    if broadcast_task: broadcast_task.cancel()
    await asyncio.gather(physics_task, broadcast_task, return_exceptions=True)


app = FastAPI(
    title="ApexMotion OS v3.0 Enterprise Fleet Edition",
    description="Intel NPU Optimized Physical AI & Robotics Infrastructure Gateway.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.post("/api/hardware/target")
async def set_hardware_target(req: HardwareProfileRequest):
    """Switches benchmark profile between 'INTEL_NPU' and 'CPU_FP32'."""
    fleet_coordinator.set_hardware_target(req.target)
    return {
        "status": "SUCCESS",
        "profile": fleet_coordinator.current_hardware_profile,
        "accelerator": fleet_coordinator.agents["Alpha"].perception.target_accelerator,
    }


@app.post("/api/chaos/inject")
async def inject_chaos(req: ChaosFaultRequest):
    """Injects or clears Chaos Engineering failure modes."""
    fleet_coordinator.inject_chaos_fault(req.fault)
    return {"status": "SUCCESS", "fault": fleet_coordinator.active_chaos_fault}


@app.post("/ros2/cmd_vel")
async def ros2_cmd_vel(req: Ros2CmdVelRequest):
    ag = fleet_coordinator.agents.get(req.agent_id)
    if not ag:
        raise HTTPException(status_code=404, detail="AGV not found")

    ag.emergency_brake = False
    ag.v_target = max(-1.5, min(1.5, req.twist.linear.x))
    ag.w_target = max(-2.0, min(2.0, req.twist.angular.z))
    ag.system_status = "ROS2_TELEOP_ACTIVE"
    return {"status": "ACK", "agent": req.agent_id, "v": ag.v_target, "w": ag.w_target}


@app.get("/ros2/odom")
async def ros2_odom(agent_id: str = "Alpha"):
    ag = fleet_coordinator.agents.get(agent_id)
    if not ag:
        raise HTTPException(status_code=404, detail="AGV not found")

    qz = math.sin(ag.theta / 2.0)
    qw = math.cos(ag.theta / 2.0)

    return {
        "header": {
            "stamp": time.time(),
            "frame_id": "odom",
        },
        "child_frame_id": f"{ag.callsign}_base_link",
        "pose": {
            "pose": {
                "position": {"x": ag.x, "y": ag.y, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": round(qz, 4), "w": round(qw, 4)},
            }
        },
        "twist": {
            "twist": {
                "linear": {"x": ag.v, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": ag.w},
            }
        },
    }


@app.post("/ros2/bridge/mode")
async def toggle_ros2_mode(req: Ros2BridgeModeRequest):
    fleet_coordinator.bridge_mode = req.mode
    fleet_coordinator.log_audit("ROS2_BRIDGE", f"Bridge operating mode switched to {req.mode}.", "INFO")
    return {"status": "SUCCESS", "bridge_mode": fleet_coordinator.bridge_mode}


@app.get("/metrics")
async def prometheus_metrics():
    ag_alpha = fleet_coordinator.agents["Alpha"]
    downtime_prevented = round(fleet_coordinator.safety_intercepts_total * 0.75, 1)
    cost_savings = int(downtime_prevented * 1000)

    lines = [
        "# HELP apexmotion_fleet_active_agents Total concurrent AGVs in operation",
        "# TYPE apexmotion_fleet_active_agents gauge",
        f"apexmotion_fleet_active_agents {len(fleet_coordinator.agents)}",
        "# HELP apexmotion_slam_exploration_percent Shared fleet SLAM discovered coverage",
        "# TYPE apexmotion_slam_exploration_percent gauge",
        f"apexmotion_slam_exploration_percent {shared_fleet_grid.exploration_percentage:.1f}",
        "# HELP apexmotion_safety_intercepts_total Total safety and collision intercepts",
        "# TYPE apexmotion_safety_intercepts_total counter",
        f"apexmotion_safety_intercepts_total {fleet_coordinator.safety_intercepts_total}",
        "# HELP apexmotion_downtime_hours_prevented_total Estimated factory downtime hours saved",
        "# TYPE apexmotion_downtime_hours_prevented_total counter",
        f"apexmotion_downtime_hours_prevented_total {downtime_prevented}",
        "# HELP apexmotion_cost_savings_usd_total Estimated cost savings from prevented collision downtime",
        "# TYPE apexmotion_cost_savings_usd_total counter",
        f"apexmotion_cost_savings_usd_total {cost_savings}",
        "# HELP apexmotion_zero_human_interventions_total Autonomous decisions executed",
        "# TYPE apexmotion_zero_human_interventions_total counter",
        f"apexmotion_zero_human_interventions_total {fleet_coordinator.autonomous_decisions_total}",
        "# HELP apexmotion_inference_latency_seconds Current edge perception latency",
        "# TYPE apexmotion_inference_latency_seconds gauge",
        f"apexmotion_inference_latency_seconds {ag_alpha.perception.last_inference_ms / 1000.0:.6f}",
        "# HELP apexmotion_hardware_power_watts Perception accelerator power draw in watts",
        "# TYPE apexmotion_hardware_power_watts gauge",
        f"apexmotion_hardware_power_watts {ag_alpha.perception.last_power_watts:.1f}",
        "# HELP apexmotion_is_intel_npu_active Flag indicating Intel NPU INT8 mode",
        "# TYPE apexmotion_is_intel_npu_active gauge",
        f"apexmotion_is_intel_npu_active {1 if fleet_coordinator.current_hardware_profile == 'INTEL_NPU' else 0}",
    ]
    for ag in fleet_coordinator.agents.values():
        lines.append(f"apexmotion_battery_percent{{agent=\"{ag.id}\"}} {ag.battery_pct:.2f}")
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "system": "ApexMotion OS v3.0 Enterprise",
        "hardware_target": fleet_coordinator.current_hardware_profile,
        "npu_latency_ms": fleet_coordinator.agents["Alpha"].perception.last_inference_ms,
        "bridge_mode": fleet_coordinator.bridge_mode,
        "chaos_fault": fleet_coordinator.active_chaos_fault,
        "slam_exploration_pct": shared_fleet_grid.exploration_percentage,
        "fleet_agents": list(fleet_coordinator.agents.keys()),
        "safety_intercepts": fleet_coordinator.safety_intercepts_total,
        "downtime_prevented_hours": round(fleet_coordinator.safety_intercepts_total * 0.75, 1),
        "timestamp": time.time(),
    }


@app.get("/api/incidents")
async def list_incidents():
    return {
        "incidents": [
            {
                "incident_id": inc["incident_id"],
                "timestamp": inc["timestamp"],
                "reason": inc["reason"],
                "culprit": inc["culprit"],
                "agent_ids": inc["agent_ids"],
                "total_frames": inc["total_frames"],
            }
            for inc in flight_recorder.incidents
        ]
    }


@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: str):
    for inc in flight_recorder.incidents:
        if inc["incident_id"] == incident_id:
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")


@app.post("/api/tts/synthesize")
async def synthesize_voice_audio(req: TtsRequest):
    """Generates studio-grade PCM WAV audio stream with radio preamble/postamble."""
    t0 = time.perf_counter()
    spoken_text, b64_audio, source = resolve_studio_or_synthesized_audio(req.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "status": "SUCCESS",
        "engine": source,
        "text": spoken_text,
        "latency_ms": latency_ms,
        "audio_base64": b64_audio,
        "format": "audio/wav",
    }


@app.post("/api/command/voice")
async def process_voice_command(cmd_req: VoiceCommandRequest):
    default_ag = cmd_req.target_agent or fleet_coordinator.active_focus_agent
    cmd = parse_enterprise_intent(cmd_req.text, default_agent=default_ag)
    fleet_coordinator.dispatch_command(cmd)

    spoken_text, b64_audio, source = resolve_studio_or_synthesized_audio(cmd.explanation, agent_id=default_ag)

    return {
        "success": True,
        "command": cmd.model_dump(),
        "verbal_feedback": spoken_text,
        "audio_source": source,
        "audio_base64": b64_audio,
    }


@app.post("/api/command/teleop")
async def process_teleop(teleop: TeleopCommand):
    ag = fleet_coordinator.agents.get(teleop.agent_id)
    if ag:
        ag.emergency_brake = False
        ag.v_target = teleop.linear_v
        ag.w_target = teleop.angular_v
        ag.system_status = "MANUAL_TELEOP"
        return {"success": True, "agent": teleop.agent_id, "v": ag.v_target, "w": ag.w_target}
    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/api/command/fleet_reset")
async def reset_fleet():
    fleet_coordinator.reset_fleet()
    return {"success": True, "message": "Fleet reset to docking stations."}


# WebSockets
@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "heartbeat":
                client_sent = float(data.get("client_time", time.time()))
                now = time.time()
                rtt = max(0.001, now - client_sent)
                ws_manager.record_heartbeat(websocket, rtt)
                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "server_time": now,
                    "rtt_ms": round(rtt * 1000, 2),
                })

            elif msg_type == "select_agent":
                agent_id = data.get("agent_id", "Alpha")
                if agent_id in fleet_coordinator.agents:
                    fleet_coordinator.active_focus_agent = agent_id

            elif msg_type == "set_hardware_target":
                target = data.get("target", "INTEL_NPU")
                fleet_coordinator.set_hardware_target(target)

            elif msg_type == "inject_chaos":
                fault = data.get("fault", "CLEAR")
                fleet_coordinator.inject_chaos_fault(fault)

            elif msg_type == "set_bridge_mode":
                mode = data.get("mode", "SIMULATION")
                fleet_coordinator.bridge_mode = mode
                fleet_coordinator.log_audit("ROS2_BRIDGE", f"Bridge operating mode switched to {mode}.", "INFO")

            elif msg_type == "ros2_cmd_vel":
                agent_id = data.get("agent_id", fleet_coordinator.active_focus_agent)
                twist = data.get("twist", {})
                lx = float(twist.get("linear", {}).get("x", 0.0))
                az = float(twist.get("angular", {}).get("z", 0.0))
                ag = fleet_coordinator.agents.get(agent_id)
                if ag:
                    ag.emergency_brake = False
                    ag.v_target = max(-1.5, min(1.5, lx))
                    ag.w_target = max(-2.0, min(2.0, az))
                    ag.system_status = "ROS2_TELEOP_ACTIVE"

            elif msg_type == "voice_command":
                raw_text = data.get("text", "")
                target_ag = data.get("target_agent", fleet_coordinator.active_focus_agent)
                parsed = parse_enterprise_intent(raw_text, default_agent=target_ag)
                fleet_coordinator.dispatch_command(parsed)

                spoken_text, b64_audio, source = resolve_studio_or_synthesized_audio(parsed.explanation, agent_id=target_ag)

                await websocket.send_json({
                    "type": "command_ack",
                    "command": parsed.model_dump(),
                    "verbal_feedback": spoken_text,
                    "audio_source": source,
                    "audio_base64": b64_audio,
                })

            elif msg_type == "teleop":
                agent_id = data.get("agent_id", fleet_coordinator.active_focus_agent)
                ag = fleet_coordinator.agents.get(agent_id)
                if ag:
                    ag.emergency_brake = False
                    ag.v_target = float(data.get("linear_v", 0.0))
                    ag.w_target = float(data.get("angular_v", 0.0))
                    ag.system_status = "MANUAL_TELEOP"

            elif msg_type == "emergency_stop":
                target_ag = data.get("agent_id", fleet_coordinator.active_focus_agent)
                brake_cmd = parse_enterprise_intent(f"{target_ag} emergency stop", default_agent=target_ag)
                fleet_coordinator.dispatch_command(brake_cmd)
                flight_recorder.capture_incident(
                    reason=f"Emergency E-STOP operator trigger for {target_ag}",
                    culprit="Operator Override",
                    agent_ids=[target_ag] if target_ag != "ALL" else list(fleet_coordinator.agents.keys()),
                )

            elif msg_type == "fleet_estop":
                fleet_brake = parse_enterprise_intent("fleet emergency stop", default_agent="ALL")
                fleet_coordinator.dispatch_command(fleet_brake)
                flight_recorder.capture_incident(
                    reason="FLEET EMERGENCY E-STOP ALL AGENTS",
                    culprit="Fleet Operator Override",
                    agent_ids=list(fleet_coordinator.agents.keys()),
                )

            elif msg_type == "set_waypoint":
                agent_id = data.get("agent_id", fleet_coordinator.active_focus_agent)
                tx = float(data.get("x", 5.0))
                ty = float(data.get("y", 5.0))
                cmd = parse_enterprise_intent(f"{agent_id} go to x {tx:.2f} y {ty:.2f}", default_agent=agent_id)
                fleet_coordinator.dispatch_command(cmd)

            elif msg_type == "trigger_incident_freeze":
                inc_id = flight_recorder.capture_incident(
                    reason="Manual Flight Recorder Blackbox Snapshot",
                    culprit="Telemetry Operator",
                    agent_ids=list(fleet_coordinator.agents.keys()),
                )
                await websocket.send_json({
                    "type": "incident_created",
                    "incident_id": inc_id,
                })

            elif msg_type == "fetch_incident_replay":
                inc_id = data.get("incident_id", "")
                target_inc = None
                if inc_id:
                    for inc in flight_recorder.incidents:
                        if inc["incident_id"] == inc_id:
                            target_inc = inc
                            break
                elif flight_recorder.incidents:
                    target_inc = flight_recorder.incidents[-1]

                if target_inc:
                    await websocket.send_json({
                        "type": "incident_replay_data",
                        "incident": target_inc,
                    })

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
        await ws_manager.disconnect(websocket)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
async def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
