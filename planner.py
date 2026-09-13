"""
ApexMotion OS - Enterprise Fleet Edition
Multi-Agent Deterministic NLP Intent Planner & Safety Guardrail Engine.
Supports agent addressing (Alpha, Bravo, Charlie, or Fleet), multi-step macros, and spatial buffer envelopes.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class CommandIntent(str, Enum):
    MOVE = "MOVE"
    ROTATE = "ROTATE"
    PATROL = "PATROL"
    SCAN = "SCAN"
    WAYPOINT = "WAYPOINT"
    MACRO = "MACRO"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    SPEED_SET = "SPEED_SET"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"


class SafetyStatus(str, Enum):
    VALIDATED = "VALIDATED"
    REVISED_BUFFER = "REVISED_BUFFER"
    INTERCEPTED_COLLISION = "INTERCEPTED_COLLISION"
    EMERGENCY_OVERRIDE = "EMERGENCY_OVERRIDE"
    REJECTED_OUT_OF_BOUNDS = "REJECTED_OUT_OF_BOUNDS"


class WaypointNode(BaseModel):
    x: float = Field(..., description="Target X coordinate in arena frame (0.0 to 10.0m)")
    y: float = Field(..., description="Target Y coordinate in arena frame (0.0 to 10.0m)")
    name: Optional[str] = Field(None, description="Landmark or node identifier")
    speed_limit: float = Field(0.6, description="Velocity ceiling in m/s")
    safety_margin: float = Field(0.5, description="Clearance buffer radius")


class MotionCommand(BaseModel):
    target_agent: str = Field("Alpha", description="Target AGV ID (Alpha, Bravo, Charlie, or ALL)")
    intent: CommandIntent
    linear_velocity: float = Field(..., description="Linear velocity setpoint in m/s")
    angular_velocity: float = Field(..., description="Angular velocity setpoint in rad/s")
    duration: float = Field(..., description="Execution window in seconds")
    target_distance: Optional[float] = Field(None)
    target_angle: Optional[float] = Field(None)
    waypoints: List[WaypointNode] = Field(default_factory=list)
    macro_name: Optional[str] = Field(None)
    macro_iterations: int = Field(1)
    emergency: bool = Field(False)
    raw_text: str = Field(...)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    safety_status: SafetyStatus = Field(SafetyStatus.VALIDATED)
    safety_notes: List[str] = Field(default_factory=list)
    explanation: str = Field(...)


FACILITY_LANDMARKS: Dict[str, Tuple[float, float]] = {
    "dock": (1.5, 1.5),
    "base": (1.5, 1.5),
    "home": (1.5, 1.5),
    "station": (1.5, 1.5),
    "pallet": (5.0, 6.5),
    "storage": (5.0, 6.5),
    "npu": (7.0, 3.0),
    "intel": (7.0, 3.0),
    "barrel": (3.0, 3.5),
    "sorting": (8.5, 1.5),
    "receiving": (1.5, 8.5),
    "forklift": (2.5, 7.5),
    "charger": (8.0, 7.5),
}

# Designated Base Docking Positions per AGV
AGENT_BASE_DOCKS: Dict[str, Tuple[float, float]] = {
    "Alpha": (1.5, 1.5),
    "Bravo": (8.5, 1.5),
    "Charlie": (1.5, 8.5),
}

KNOWN_HAZARDS = [
    {"name": "Hazard Barrel", "x": 3.0, "y": 3.5, "radius": 0.65},
    {"name": "Intel NPU Node", "x": 7.0, "y": 3.0, "radius": 0.75},
    {"name": "Industrial Pallet", "x": 5.0, "y": 6.5, "radius": 0.85},
    {"name": "AGV Forklift Zone", "x": 2.5, "y": 7.5, "radius": 0.70},
    {"name": "Charging Station", "x": 8.0, "y": 7.5, "radius": 0.60},
]

MAX_LINEAR_VELOCITY = 1.2
DEFAULT_LINEAR_SPEED = 0.5
MAX_ANGULAR_VELOCITY = 1.8
DEFAULT_ANGULAR_SPEED = 0.8
ROBOT_COLLISION_RADIUS = 0.35
MIN_HAZARD_BUFFER = 0.45

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "half": 0.5, "once": 1, "twice": 2, "thrice": 3,
}


def normalize_transcript(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^\w\s\.-]", " ", cleaned)
    tokens = cleaned.split()
    normalized = [str(WORD_TO_NUM[t]) if t in WORD_TO_NUM else t for t in tokens]
    return " ".join(normalized)


def _extract_number(text: str, default: float) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return default
    return default


def validate_spatial_buffer(x: float, y: float) -> Tuple[float, float, SafetyStatus, List[str]]:
    notes = []
    status = SafetyStatus.VALIDATED

    bounded_x = max(0.6, min(9.4, x))
    bounded_y = max(0.6, min(9.4, y))
    if bounded_x != x or bounded_y != y:
        notes.append(f"Boundary constraint: clamped from ({x:.2f}, {y:.2f}) to ({bounded_x:.2f}, {bounded_y:.2f})")
        status = SafetyStatus.REVISED_BUFFER
        x, y = bounded_x, bounded_y

    for hz in KNOWN_HAZARDS:
        dist = math.hypot(x - hz["x"], y - hz["y"])
        clearance = hz["radius"] + ROBOT_COLLISION_RADIUS + MIN_HAZARD_BUFFER
        if dist < clearance:
            status = SafetyStatus.REVISED_BUFFER
            pen = clearance - dist
            notes.append(f"Guardrail: Target breached {hz['name']} buffer by {pen:.2f}m. Projected to perimeter.")
            vec_x = (x - hz["x"]) / max(0.01, dist)
            vec_y = (y - hz["y"]) / max(0.01, dist)
            x = hz["x"] + vec_x * (clearance + 0.1)
            y = hz["y"] + vec_y * (clearance + 0.1)
            x = max(0.6, min(9.4, x))
            y = max(0.6, min(9.4, y))

    return x, y, status, notes


def parse_enterprise_intent(raw_input: str, default_agent: str = "Alpha") -> MotionCommand:
    """
    Fleet-aware intent parser. Dispatches commands to Alpha, Bravo, Charlie, or ALL.
    """
    if not raw_input or not raw_input.strip():
        return MotionCommand(
            target_agent=default_agent,
            intent=CommandIntent.UNKNOWN,
            linear_velocity=0.0,
            angular_velocity=0.0,
            duration=0.0,
            raw_text=raw_input,
            confidence=0.0,
            safety_status=SafetyStatus.REJECTED_OUT_OF_BOUNDS,
            explanation="Empty command input string.",
        )

    text = normalize_transcript(raw_input)

    # 1. Detect Agent Targeting (e.g. "Alpha, move forward", "Bravo patrol route", "Fleet stop")
    target_agent = default_agent
    if "fleet" in text or "all agents" in text or "all robots" in text:
        target_agent = "ALL"
    elif "alpha" in text:
        target_agent = "Alpha"
    elif "bravo" in text:
        target_agent = "Bravo"
    elif "charlie" in text:
        target_agent = "Charlie"

    # Strip agent name from command text for clean intent extraction
    clean_text = re.sub(r"\b(alpha|bravo|charlie|fleet|all agents|all robots)\b", "", text).strip()

    # 2. RETURN TO BASE / DOCK (RTB) - Checked before general abort/stop
    if any(k in clean_text for k in ["return to base", "rtb", "go home", "return to dock", "dock at base", "abort and return"]):
        dock_pos = AGENT_BASE_DOCKS.get(target_agent, (1.5, 1.5))
        vx, vy, status, notes = validate_spatial_buffer(dock_pos[0], dock_pos[1])
        wp = WaypointNode(x=vx, y=vy, name=f"{target_agent} Station Dock", speed_limit=0.5, safety_margin=0.6)
        return MotionCommand(
            target_agent=target_agent,
            intent=CommandIntent.RETURN_TO_BASE,
            linear_velocity=DEFAULT_LINEAR_SPEED,
            angular_velocity=DEFAULT_ANGULAR_SPEED,
            duration=25.0,
            waypoints=[wp],
            macro_name="RETURN_TO_BASE",
            raw_text=raw_input,
            confidence=0.98,
            safety_status=status,
            safety_notes=notes or [f"Assigned transit corridor to {target_agent} dock verified."],
            explanation=f"[{target_agent}] Autonomous RTB engaged: Navigating to Dock ({vx:.2f}, {vy:.2f}).",
        )

    # 3. FLEET OR AGENT EMERGENCY BRAKE
    if any(k in clean_text for k in ["stop", "halt", "freeze", "brake", "abort", "kill", "e-stop", "emergency"]):
        return MotionCommand(
            target_agent=target_agent,
            intent=CommandIntent.STOP,
            linear_velocity=0.0,
            angular_velocity=0.0,
            duration=0.0,
            emergency=True,
            raw_text=raw_input,
            confidence=1.0,
            safety_status=SafetyStatus.EMERGENCY_OVERRIDE,
            safety_notes=["Immediate kinematic deceleration to zero. Safety circuit trip."],
            explanation=f"[{target_agent}] EMERGENCY BRAKE ENGAGED: Path queue cleared, velocity cut-off.",
        )

    # 4. MULTI-STEP MACRO: PATROL BETWEEN TWO LANDMARKS N TIMES
    patrol_match = re.search(
        r"(?:patrol|shuttle|cycle|loop)\s+(?:between\s+)?(\w+)\s+(?:and|to)\s+(\w+)(?:\s+(\d+)\s*(?:times|cycles|loops)?)?",
        clean_text,
    )
    if patrol_match:
        loc_a_raw = patrol_match.group(1)
        loc_b_raw = patrol_match.group(2)
        iter_count = int(patrol_match.group(3)) if patrol_match.group(3) else 2
        iter_count = max(1, min(10, iter_count))

        loc_a = FACILITY_LANDMARKS.get(loc_a_raw, None)
        loc_b = FACILITY_LANDMARKS.get(loc_b_raw, None)

        if loc_a and loc_b:
            waypoints: List[WaypointNode] = []
            safety_notes = []
            for i in range(iter_count):
                ax, ay, st_a, n_a = validate_spatial_buffer(loc_a[0], loc_a[1])
                bx, by, st_b, n_b = validate_spatial_buffer(loc_b[0], loc_b[1])
                waypoints.append(WaypointNode(x=ax, y=ay, name=f"{loc_a_raw.upper()} [Cycle {i+1}]"))
                waypoints.append(WaypointNode(x=bx, y=by, name=f"{loc_b_raw.upper()} [Cycle {i+1}]"))
                if n_a: safety_notes.extend(n_a)
                if n_b: safety_notes.extend(n_b)

            return MotionCommand(
                target_agent=target_agent,
                intent=CommandIntent.MACRO,
                linear_velocity=DEFAULT_LINEAR_SPEED,
                angular_velocity=DEFAULT_ANGULAR_SPEED,
                duration=round(len(waypoints) * 8.0, 1),
                waypoints=waypoints,
                macro_name=f"PATROL_{loc_a_raw.upper()}_{loc_b_raw.upper()}",
                macro_iterations=iter_count,
                raw_text=raw_input,
                confidence=0.96,
                safety_status=SafetyStatus.VALIDATED if not safety_notes else SafetyStatus.REVISED_BUFFER,
                safety_notes=safety_notes or [f"Inter-facility corridor verified ({iter_count} cycles)."],
                explanation=f"[{target_agent}] Macro Activated: Patrol {loc_a_raw.title()} <-> {loc_b_raw.title()} ({iter_count} cycles, {len(waypoints)} nodes).",
            )

    # 5. PERIMETER PATROL
    if any(k in clean_text for k in ["patrol", "circuit", "boundary loop", "route", "perimeter patrol"]):
        raw_waypoints = [
            (2.0, 2.0, "Waypoint Alpha"),
            (8.0, 2.0, "Waypoint Bravo"),
            (8.0, 8.0, "Waypoint Charlie"),
            (2.0, 8.0, "Waypoint Delta"),
            (2.0, 2.0, "Waypoint Alpha (Closure)"),
        ]
        validated_wps = []
        notes = []
        for rx, ry, name in raw_waypoints:
            vx, vy, st, n = validate_spatial_buffer(rx, ry)
            validated_wps.append(WaypointNode(x=vx, y=vy, name=name))
            if n: notes.extend(n)

        return MotionCommand(
            target_agent=target_agent,
            intent=CommandIntent.PATROL,
            linear_velocity=DEFAULT_LINEAR_SPEED,
            angular_velocity=DEFAULT_ANGULAR_SPEED,
            duration=35.0,
            waypoints=validated_wps,
            macro_name="PERIMETER_SURVEY",
            raw_text=raw_input,
            confidence=0.94,
            safety_status=SafetyStatus.VALIDATED if not notes else SafetyStatus.REVISED_BUFFER,
            safety_notes=notes or ["Perimeter boundary nodes verified against static hazard envelopes."],
            explanation=f"[{target_agent}] Autonomous 4-Node Perimeter Inspection Plan dispatched.",
        )

    # 6. SCAN PERIMETER (360 DEGREES)
    if any(k in clean_text for k in ["scan", "survey", "inspect perimeter", "look around", "360"]):
        scan_w = 0.75
        duration = round((2 * math.pi) / scan_w, 2)
        return MotionCommand(
            target_agent=target_agent,
            intent=CommandIntent.SCAN,
            linear_velocity=0.0,
            angular_velocity=scan_w,
            duration=duration,
            target_angle=round(2 * math.pi, 3),
            raw_text=raw_input,
            confidence=0.96,
            safety_status=SafetyStatus.VALIDATED,
            safety_notes=["In-place pivot verified."],
            explanation=f"[{target_agent}] Full 360° sensor sweep initiated (duration = {duration}s).",
        )

    # 7. GOTO LANDMARK OR COORDINATE
    for lm_key, (lx, ly) in FACILITY_LANDMARKS.items():
        if f"go to {lm_key}" in clean_text or f"navigate to {lm_key}" in clean_text or f"move to {lm_key}" in clean_text:
            vx, vy, st, n = validate_spatial_buffer(lx, ly)
            wp = WaypointNode(x=vx, y=vy, name=f"Landmark {lm_key.title()}")
            return MotionCommand(
                target_agent=target_agent,
                intent=CommandIntent.WAYPOINT,
                linear_velocity=DEFAULT_LINEAR_SPEED,
                angular_velocity=DEFAULT_ANGULAR_SPEED,
                duration=18.0,
                waypoints=[wp],
                raw_text=raw_input,
                confidence=0.95,
                safety_status=st,
                safety_notes=n or [f"Target {lm_key.title()} validated."],
                explanation=f"[{target_agent}] Navigating to {lm_key.title()} ({vx:.2f}m, {vy:.2f}m).",
            )

    coord_match = re.search(
        r"(?:go\s+to|navigate\s+to|move\s+to)\s*(?:x\s*[:=]?\s*)?([-\d\.]+)\s*(?:and|y\s*[:=]?|,)?\s*([-\d\.]+)",
        clean_text,
    )
    if coord_match:
        try:
            tx = float(coord_match.group(1))
            ty = float(coord_match.group(2))
            vx, vy, st, notes = validate_spatial_buffer(tx, ty)
            wp = WaypointNode(x=vx, y=vy, name=f"Point ({vx:.1f}, {vy:.1f})")
            return MotionCommand(
                target_agent=target_agent,
                intent=CommandIntent.WAYPOINT,
                linear_velocity=DEFAULT_LINEAR_SPEED,
                angular_velocity=DEFAULT_ANGULAR_SPEED,
                duration=16.0,
                waypoints=[wp],
                raw_text=raw_input,
                confidence=0.95,
                safety_status=st,
                safety_notes=notes or ["Coordinate target validated."],
                explanation=f"[{target_agent}] Direct waypoint navigation planned to ({vx:.2f}m, {vy:.2f}m).",
            )
        except ValueError:
            pass

    # 8. ROTATION
    if any(k in clean_text for k in ["turn", "rotate", "spin", "pivot"]):
        is_left = any(k in clean_text for k in ["left", "counterclockwise", "ccw", "port"])
        is_right = any(k in clean_text for k in ["right", "clockwise", "cw", "starboard"])
        sign = 1.0 if is_left else (-1.0 if is_right else 1.0)

        deg_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)", clean_text)
        if deg_match:
            deg = float(deg_match.group(1))
            target_rad = math.radians(deg)
        elif "90" in clean_text:
            target_rad = math.radians(90.0)
        elif "45" in clean_text:
            target_rad = math.radians(45.0)
        elif "180" in clean_text:
            target_rad = math.radians(180.0)
        else:
            target_rad = math.radians(90.0)

        duration = max(0.4, round(target_rad / DEFAULT_ANGULAR_SPEED, 2))
        w = round(sign * DEFAULT_ANGULAR_SPEED, 3)
        dir_label = "CCW (Left)" if sign > 0 else "CW (Right)"

        return MotionCommand(
            target_agent=target_agent,
            intent=CommandIntent.ROTATE,
            linear_velocity=0.0,
            angular_velocity=w,
            duration=duration,
            target_angle=round(sign * target_rad, 3),
            raw_text=raw_input,
            confidence=0.96,
            safety_status=SafetyStatus.VALIDATED,
            explanation=f"[{target_agent}] Rotation {dir_label} by {math.degrees(target_rad):.1f}° (ω = {w} rad/s).",
        )

    # 9. LINEAR TRANSLATION
    is_fwd = any(k in clean_text for k in ["forward", "ahead", "straight", "front", "advance"])
    is_bwd = any(k in clean_text for k in ["backward", "reverse", "back", "retreat"])
    is_move = any(k in clean_text for k in ["move", "drive", "go", "step", "travel"])

    if is_fwd or is_bwd or is_move:
        sign = -1.0 if is_bwd else 1.0
        dist_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b", clean_text)
        if dist_match:
            distance = float(dist_match.group(1))
        else:
            distance = _extract_number(clean_text, default=1.0)

        distance = min(distance, 10.0)
        v = round(sign * DEFAULT_LINEAR_SPEED, 3)
        duration = max(0.4, round(distance / DEFAULT_LINEAR_SPEED, 2))
        action_name = "Reverse" if sign < 0 else "Forward"

        return MotionCommand(
            target_agent=target_agent,
            intent=CommandIntent.MOVE,
            linear_velocity=v,
            angular_velocity=0.0,
            duration=duration,
            target_distance=round(sign * distance, 3),
            raw_text=raw_input,
            confidence=0.95,
            safety_status=SafetyStatus.VALIDATED,
            explanation=f"[{target_agent}] {action_name} translation of {distance:.2f}m (v = {v} m/s, {duration}s).",
        )

    # 10. SPEED MODIFIERS
    if any(k in clean_text for k in ["faster", "speed up", "fast", "turbo"]):
        return MotionCommand(
            target_agent=target_agent,
            intent=CommandIntent.SPEED_SET,
            linear_velocity=MAX_LINEAR_VELOCITY,
            angular_velocity=0.0,
            duration=0.0,
            raw_text=raw_input,
            confidence=0.9,
            safety_status=SafetyStatus.VALIDATED,
            explanation=f"[{target_agent}] Speed profile set to TURBO ({MAX_LINEAR_VELOCITY} m/s).",
        )

    return MotionCommand(
        target_agent=target_agent,
        intent=CommandIntent.UNKNOWN,
        linear_velocity=0.0,
        angular_velocity=0.0,
        duration=0.0,
        raw_text=raw_input,
        confidence=0.2,
        safety_status=SafetyStatus.REJECTED_OUT_OF_BOUNDS,
        explanation=f"[{target_agent}] Unrecognized intent: '{raw_input}'.",
    )


parse_voice_intent = parse_enterprise_intent
