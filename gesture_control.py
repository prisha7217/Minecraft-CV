"""
Minecraft Gesture Control System (v3.3)
========================================
Uses MediaPipe hand tracking + pydirectinput to play Minecraft with hand gestures.

Left hand  (gesture): gestures, hotbar (thumbs-up), jump (3 fingers up)
Right hand (mouse):   palm delta controls mouse look (fist = rest)

Gesture Priority Hierarchy:
  1. Kill Switch  (pynput 'K' / 'End' key)
  2. UI State     (Inventory — open palm; fist also closes it)
  3. Actions      (Attack / Interact — pinch with hysteresis)
  4. Jump         (Index+Middle+Ring up, Pinky down → Spacebar)
  5. Hotbar       (Thumbs-up + wrist tilt)
  6. Stop         (Fist — releases all keys; also closes inventory)
  7. Movement     (Sprint+Forward — Index+Middle up → Ctrl+W)
  8. Modifier     (Sneak — Pinky up, combinable → Shift)

Stability: 3-frame gesture smoothing, pinch hysteresis, EMA delta mouse.
Kill switch: Press 'K' or 'End' key to quit.
Run as Administrator for DirectInput to work.
"""

import cv2
import mediapipe as mp
import pydirectinput
import numpy as np
import math
import time
import sys
from collections import deque

from pynput.keyboard import Key, Listener as KeyboardListener

# ─── Configuration ────────────────────────────────────────────────────────────

pydirectinput.PAUSE = 0       # No delay between inputs
pydirectinput.FAILSAFE = False  # Don't abort on corner mouse

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Deadzone — covers entire frame (primary hand never moves mouse;
# mouse look is handled exclusively by the second hand's index finger)
DEADZONE_WIDTH = FRAME_WIDTH    # 640 — full screen width
DEADZONE_HEIGHT = FRAME_HEIGHT  # 480 — full screen height

# Primary hand mouse sensitivity (palm-based, inside deadzone = no move)
MOUSE_SENSITIVITY = 0.15

# ─── Second-Hand Mouse Configuration ─────────────────────────────────────────

MOUSE2_SENSITIVITY = 2.5        # Multiplier for finger delta → mouse pixels
MOUSE2_EMA_ALPHA   = 0.4        # EMA smoothing for deltas (0-1, lower = smoother)
MOUSE2_MAX_MOVE    = 20         # Max pixels per frame for second hand

# Pinch thresholds — hysteresis to prevent flickering
PINCH_ENTER = 0.055   # Distance to START pinch
PINCH_EXIT  = 0.08    # Distance to RELEASE pinch

# Cooldown for one-shot actions (seconds)
ACTION_COOLDOWN = 0.35

# No-hand timeout: release all keys if hand vanishes for N frames
NO_HAND_TIMEOUT_FRAMES = 8

# Gesture smoothing — require N consecutive frames of same gesture
SMOOTH_FRAMES = 3

# ─── Jump Configuration ──────────────────────────────────────────────────────

JUMP_COOLDOWN = 0.5             # Seconds between jump presses

# ─── Hotbar Scroll Configuration ─────────────────────────────────────────────

SCROLL_DEBOUNCE = 0.3           # Seconds between slot changes
SCROLL_ANGLE_MIN = -60          # Degrees — leftmost tilt → slot 1
SCROLL_ANGLE_MAX = 60           # Degrees — rightmost tilt → slot 9

# ─── Global State ─────────────────────────────────────────────────────────────

running = True                   # Kill-switch flag
pressed_keys: set[str] = set()   # Currently held keys (multi-key support)
inventory_locked = False         # True while palm is open (blocks other input)
hotbar_slot = 0                  # Current hotbar slot (0-8)

# Pinch state (for hysteresis)
pinch_thumb_index_active = False
pinch_thumb_pinky_active = False

# Jump state
last_jump_time = 0.0

# Hotbar scroll state
last_scroll_time = 0.0
prev_scroll_slot = -1

# Gesture smoothing state
gesture_buffer: deque[frozenset] = deque(maxlen=SMOOTH_FRAMES)

# Second-hand mouse state (delta-based)
mouse2_prev_x: float | None = None
mouse2_prev_y: float | None = None

# Debug values (for overlay)
debug_hand_angle = 0.0
debug_mouse2_active = False

# ─── Kill Switch (pynput listener) ────────────────────────────────────────────

def on_key_press(key):
    """Listen for kill-switch key presses in a background thread."""
    global running
    try:
        if hasattr(key, 'char') and key.char == 'k':
            running = False
        elif key == Key.end:
            running = False
    except AttributeError:
        pass


def start_kill_switch():
    """Start a daemon thread that listens for the kill-switch."""
    listener = KeyboardListener(on_press=on_key_press)
    listener.daemon = True
    listener.start()
    return listener


# ─── Key Management (set-based) ──────────────────────────────────────────────

def hold_key(key_name: str):
    """Press and hold a key (only sends keyDown if not already held)."""
    if key_name not in pressed_keys:
        pydirectinput.keyDown(key_name)
        pressed_keys.add(key_name)


def release_key(key_name: str):
    """Release a key (only sends keyUp if currently held)."""
    if key_name in pressed_keys:
        pydirectinput.keyUp(key_name)
        pressed_keys.discard(key_name)


def release_all():
    """Release every key that is currently held."""
    for key_name in list(pressed_keys):
        pydirectinput.keyUp(key_name)
    pressed_keys.clear()


def sync_keys(desired: set[str]):
    """
    Sync the held-key set to exactly match `desired`.
    Only sends keyDown/keyUp for keys that actually changed.
    """
    to_release = pressed_keys - desired
    to_press = desired - pressed_keys

    # Nothing changed → skip entirely (reduces DirectInput spam)
    if not to_release and not to_press:
        return

    for k in to_release:
        pydirectinput.keyUp(k)
    for k in to_press:
        pydirectinput.keyDown(k)

    pressed_keys.clear()
    pressed_keys.update(desired)


# ─── Finger State Helpers ────────────────────────────────────────────────────

def is_finger_up(landmarks, tip_id, pip_id, hand_label):
    """Check if a finger is extended."""
    tip = landmarks[tip_id]
    pip = landmarks[pip_id]

    if tip_id == 4:  # Thumb — compare x instead of y
        if hand_label == "Right":
            return tip.x < pip.x  # Mirrored camera
        else:
            return tip.x > pip.x
    else:
        return tip.y < pip.y


def get_finger_states(landmarks, hand_label):
    """Return [thumb, index, middle, ring, pinky] booleans."""
    finger_ids = [
        (4, 3),    # Thumb: tip vs IP
        (8, 6),    # Index: tip vs PIP
        (12, 10),  # Middle: tip vs PIP
        (16, 14),  # Ring: tip vs PIP
        (20, 18),  # Pinky: tip vs PIP
    ]
    return [is_finger_up(landmarks, t, p, hand_label) for t, p in finger_ids]


def landmark_distance(landmarks, id1, id2):
    """Euclidean distance between two normalised landmarks."""
    a, b = landmarks[id1], landmarks[id2]
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ─── Pinch Detection with Hysteresis ─────────────────────────────────────────

def check_pinch_hysteresis(dist, is_active):
    """
    Returns True/False for pinch state.
    Uses different thresholds for entering vs exiting pinch to prevent flicker.
    """
    if is_active:
        return dist < PINCH_EXIT   # Stay active until distance exceeds exit
    else:
        return dist < PINCH_ENTER  # Only activate below tight threshold


# ─── Jump Detection (Gesture-Based) ──────────────────────────────────────────
# Jump is now triggered by raising 3 fingers (index+middle+ring up, pinky down)
# on the gesture hand. Much more reliable than jolt detection.


# ─── Angle-Based Hotbar Scroll (only during thumbs-up) ───────────────────────

def compute_hand_angle(landmarks):
    """
    Compute the tilt angle of the hand using atan2 between
    the wrist (landmark 0) and middle-finger MCP (landmark 9).
    Returns angle in degrees.
    """
    global debug_hand_angle

    wrist = landmarks[0]
    mcp = landmarks[9]

    dx = mcp.x - wrist.x
    dy = mcp.y - wrist.y
    angle_rad = math.atan2(dx, -dy)  # -dy because y is flipped
    angle_deg = math.degrees(angle_rad)
    debug_hand_angle = angle_deg
    return angle_deg


def handle_hotbar_scroll(landmarks):
    """
    Map the hand tilt angle to a hotbar slot (1-9).
    Only fires when the slot changes and debounce has elapsed.
    Returns the new slot number (1-9) or 0 if no change.
    """
    global hotbar_slot, last_scroll_time, prev_scroll_slot

    angle = compute_hand_angle(landmarks)
    now = time.time()

    # Clamp angle to configured range
    clamped = max(float(SCROLL_ANGLE_MIN), min(float(SCROLL_ANGLE_MAX), angle))

    # Map angle to slot 0-8
    normalised = (clamped - SCROLL_ANGLE_MIN) / (SCROLL_ANGLE_MAX - SCROLL_ANGLE_MIN)
    slot = int(normalised * 8.99)  # 0..8
    slot = max(0, min(8, slot))

    # Only fire if slot changed and debounce passed
    if slot != prev_scroll_slot and (now - last_scroll_time) > SCROLL_DEBOUNCE:
        hotbar_slot = slot
        pydirectinput.press(str(slot + 1))
        last_scroll_time = now
        prev_scroll_slot = slot
        return slot + 1

    return 0


# ─── Gesture Smoothing ───────────────────────────────────────────────────────

def smooth_gesture(raw_actions: set[str]) -> set[str]:
    """
    Push the raw gesture into a rolling buffer.
    Only output a gesture if it has been stable for SMOOTH_FRAMES consecutive
    frames.  This eliminates single-frame misfires that cause key toggling.
    """
    frozen = frozenset(raw_actions)
    gesture_buffer.append(frozen)

    if len(gesture_buffer) < SMOOTH_FRAMES:
        return set()  # Not enough data yet — output nothing

    # If all frames in the buffer agree → output that gesture
    if all(g == frozen for g in gesture_buffer):
        return set(frozen)

    # Buffer disagrees → hold the PREVIOUS stable state (last agreed frame)
    # This keeps the current keys held instead of releasing them
    return set(gesture_buffer[-2]) if len(gesture_buffer) >= 2 else set()


# ─── Hierarchical Gesture Classification ─────────────────────────────────────
#
#  Priority:  UI → Pinch-Actions → Hotbar(thumbs-up) → Stop → Movement → Modifier
#  Jump is independent (velocity-based, not gesture-based).
# ──────────────────────────────────────────────────────────────────────────────

def classify_gestures(landmarks, hand_label):
    """
    Classify the hand pose using hierarchical priority.
    Returns a set of active action tags and the finger states list.
    """
    global inventory_locked, pinch_thumb_index_active, pinch_thumb_pinky_active

    fingers = get_finger_states(landmarks, hand_label)
    thumb, index, middle, ring, pinky = fingers
    actions: set[str] = set()

    # ── Priority 1: UI / Menu — Inventory (all 5 fingers extended) ──
    if all(fingers):
        actions.add("inventory")
        inventory_locked = True
        return actions, fingers

    # If inventory was locked, check if fist → close inventory
    if inventory_locked:
        # Fist (all 4 main fingers down) closes inventory
        if not index and not middle and not ring and not pinky:
            inventory_locked = False
            actions.add("close_inventory")
            return actions, fingers
        if sum(fingers) <= 2:
            inventory_locked = False
        else:
            return {"inventory_lock"}, fingers

    # ── Priority 2: Pinch actions (with hysteresis) ──
    thumb_index_dist = landmark_distance(landmarks, 4, 8)

    pinch_thumb_index_active = check_pinch_hysteresis(
        thumb_index_dist, pinch_thumb_index_active
    )

    if pinch_thumb_index_active:
        actions.add("attack")
        return actions, fingers

    # ── Priority 3: Interact — 4 fingers up (index+middle+ring+pinky, thumb down) ──
    if not thumb and index and middle and ring and pinky:
        actions.add("interact")
        return actions, fingers

    # ── Priority 4: Sprint+Jump (HELD) — 3 fingers up (index+middle+ring, pinky down) ──
    if index and middle and ring and not pinky:
        actions.add("sprint_jump")
        return actions, fingers

    # ── Priority 5: Jump (one-shot) — Thumbs-up only ──
    if thumb and not index and not middle and not ring and not pinky:
        actions.add("jump")
        return actions, fingers

    # ── Priority 6: Stop — Fist (all 4 main fingers curled, thumb can be anything) ──
    if not index and not middle and not ring and not pinky:
        actions.add("stop")
        return actions, fingers

    # ── Priority 7: Movement — Sprint+Forward (Index+Middle up, Ring+Pinky down) ──
    if index and middle and not ring and not pinky:
        actions.add("forward")

    # ── Priority 8: Modifier — Sneak (Pinky up, combinable) ──
    if pinky:
        actions.add("sneak")

    return actions, fingers


# ─── Execute Inputs ──────────────────────────────────────────────────────────

def execute_actions(actions: set[str], prev_actions: set[str], last_oneshot_time: float):
    """
    Translate action tags into pydirectinput calls.
    Toggle keys managed via sync_keys(); one-shot actions fire on edge.
    """
    global last_jump_time
    now = time.time()

    # ── Build the desired set of held keys ──
    desired_keys: set[str] = set()

    if "forward" in actions:
        desired_keys.add("w")
        desired_keys.add("ctrl")   # Sprint = Ctrl+W

    if "sprint_jump" in actions:
        desired_keys.add("w")
        desired_keys.add("ctrl")
        desired_keys.add("space")  # Hold all three for sprint-jump

    if "sneak" in actions:
        desired_keys.add("shift")

    # ── Sync toggle keys ──
    sync_keys(desired_keys)

    # ── Stop — release everything ──
    if "stop" in actions:
        release_all()
        return last_oneshot_time

    # ── One-shot actions (fire once on edge) ──

    if "attack" in actions and "attack" not in prev_actions:
        if (now - last_oneshot_time) > ACTION_COOLDOWN:
            pydirectinput.click(button="left")
            last_oneshot_time = now

    if "interact" in actions and "interact" not in prev_actions:
        if (now - last_oneshot_time) > ACTION_COOLDOWN:
            pydirectinput.click(button="right")
            last_oneshot_time = now

    if "inventory" in actions and "inventory" not in prev_actions:
        if (now - last_oneshot_time) > ACTION_COOLDOWN:
            release_all()
            pydirectinput.press("e")
            last_oneshot_time = now

    if "close_inventory" in actions and "close_inventory" not in prev_actions:
        if (now - last_oneshot_time) > ACTION_COOLDOWN:
            release_all()
            pydirectinput.press("e")  # Press E again to close
            last_oneshot_time = now

    if "jump" in actions and "jump" not in prev_actions:
        if (now - last_jump_time) > JUMP_COOLDOWN:
            pydirectinput.keyDown("space")
            time.sleep(0.05)
            pydirectinput.keyUp("space")
            last_jump_time = now

    return last_oneshot_time


# ─── Mouse Look (Relative Movement with Deadzone) ────────────────────────────

def handle_mouse_look(landmarks, frame_w, frame_h):
    """Move the mouse based on palm position relative to the deadzone."""
    palm = landmarks[9]  # Middle-finger MCP
    px = int(palm.x * frame_w)
    py = int(palm.y * frame_h)

    cx, cy = frame_w // 2, frame_h // 2
    half_w, half_h = DEADZONE_WIDTH // 2, DEADZONE_HEIGHT // 2

    dx = px - cx
    dy = py - cy
    outside = abs(dx) > half_w or abs(dy) > half_h

    if outside:
        move_x = int(dx * MOUSE_SENSITIVITY)
        move_y = int(dy * MOUSE_SENSITIVITY)
        move_x = max(-30, min(30, move_x))
        move_y = max(-30, min(30, move_y))
        pydirectinput.moveRel(move_x, move_y, _pause=False)

    return outside


# ─── Second-Hand Mouse (Delta-Based, EMA Smoothed) ───────────────────────────

def handle_mouse2(landmarks, hand_label, frame_w, frame_h):
    """
    Delta-based mouse: only finger MOVEMENT between frames moves the mouse.
    When finger is still → mouse is still (no drift/spinning).
    Closed fist = rest (no mouse movement, resets tracking).
    """
    global mouse2_prev_x, mouse2_prev_y, debug_mouse2_active

    # Check if fist (all 4 main fingers down) → rest, no mouse move
    fingers = get_finger_states(landmarks, hand_label)
    thumb, index, middle, ring, pinky = fingers
    if not index and not middle and not ring and not pinky:
        # Fist = rest — reset so no jump when hand reopens
        debug_mouse2_active = False
        mouse2_prev_x = None
        mouse2_prev_y = None
        return

    debug_mouse2_active = True

    # Track index finger tip position
    idx_tip = landmarks[8]
    raw_x = idx_tip.x * frame_w
    raw_y = idx_tip.y * frame_h

    # First frame → just store position, no movement
    if mouse2_prev_x is None:
        mouse2_prev_x = raw_x
        mouse2_prev_y = raw_y
        return

    # Compute delta (how much finger moved since last frame)
    dx = raw_x - mouse2_prev_x
    dy = raw_y - mouse2_prev_y

    # Update previous position (with EMA for smoothness)
    mouse2_prev_x = MOUSE2_EMA_ALPHA * raw_x + (1 - MOUSE2_EMA_ALPHA) * mouse2_prev_x
    mouse2_prev_y = MOUSE2_EMA_ALPHA * raw_y + (1 - MOUSE2_EMA_ALPHA) * mouse2_prev_y

    # Apply sensitivity and clamp
    move_x = int(dx * MOUSE2_SENSITIVITY)
    move_y = int(dy * MOUSE2_SENSITIVITY)
    move_x = max(-MOUSE2_MAX_MOVE, min(MOUSE2_MAX_MOVE, move_x))
    move_y = max(-MOUSE2_MAX_MOVE, min(MOUSE2_MAX_MOVE, move_y))

    # Only move if there's meaningful delta (deadzone to prevent micro-jitter)
    if abs(move_x) > 0 or abs(move_y) > 0:
        pydirectinput.moveRel(move_x, move_y, _pause=False)


# ─── Action → Display Info ───────────────────────────────────────────────────

ACTION_DISPLAY = {
    "forward":         ("SPRINT + FORWARD", "Ctrl+W",      (0, 255, 255)),
    "stop":            ("STOP",             "Release All",  (100, 100, 255)),
    "sneak":           ("SNEAK",            "Shift",        (200, 100, 255)),
    "attack":          ("ATTACK",           "Left Click",   (0, 100, 255)),
    "interact":        ("INTERACT",         "Right Click",  (0, 200, 100)),
    "inventory":       ("INVENTORY",        "E",            (255, 200, 0)),
    "inventory_lock":  ("INVENTORY LOCK",   "(locked)",     (100, 100, 0)),
    "close_inventory": ("CLOSE INVENTORY",  "E (close)",    (200, 150, 0)),
    "hotbar":          ("HOTBAR MODE",      "Tilt L/R",     (255, 150, 50)),
    "jump":            ("JUMP",             "Space",        (50, 255, 50)),
}

GESTURE_REFERENCE = [
    ("INVENTORY",  "Open palm (5 up)",     (255, 200, 0)),
    ("CLOSE INV",  "Fist while inv open",  (200, 150, 0)),
    ("SPRINT+FWD", "Index+Middle up",      (0, 255, 255)),
    ("STOP",       "Fist (all curled)",    (100, 100, 255)),
    ("SNEAK",      "Pinky up (combinable)",(200, 100, 255)),
    ("ATTACK",     "Thumb-Index pinch",    (0, 100, 255)),
    ("INTERACT",   "Thumb-Ring pinch",     (0, 200, 100)),
    ("HOTBAR",     "Thumbs-up + tilt",     (255, 150, 50)),
    ("JUMP",       "3 fingers up",         (50, 255, 50)),
    ("MOUSE",      "R hand palm (delta)",  (180, 180, 255)),
]

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


# ─── Visual Overlay ──────────────────────────────────────────────────────────

def draw_hand_bbox(frame, hand_landmarks, fw, fh):
    """Draw a bounding box around the detected hand."""
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    pad = 15
    x1 = max(0, int(min(xs) * fw) - pad)
    y1 = max(0, int(min(ys) * fh) - pad)
    x2 = min(fw, int(max(xs) * fw) + pad)
    y2 = min(fh, int(max(ys) * fh) + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)


def draw_overlay(frame, actions, outside_deadzone, fps, finger_states=None, jumped=False):
    """Draw debugging overlay with stability info."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    hw, hh = DEADZONE_WIDTH // 2, DEADZONE_HEIGHT // 2

    # ── Deadzone box ──
    dz_colour = (0, 0, 255) if outside_deadzone else (0, 255, 0)
    cv2.rectangle(frame, (cx - hw, cy - hh), (cx + hw, cy + hh), dz_colour, 2)
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), dz_colour, 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), dz_colour, 1)
    if outside_deadzone:
        cv2.putText(frame, "LOOK", (cx - 20, cy - hh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ── Active actions + key labels (top-centre) ──
    y_offset = 40
    for act in sorted(actions):
        info = ACTION_DISPLAY.get(act)
        if info:
            name, key, colour = info
            label = f"{name}  [{key}]"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            tx = (w - ts[0]) // 2
            cv2.putText(frame, label, (tx + 1, y_offset + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, label, (tx, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            y_offset += 30

    # ── Jump flash ──
    if jumped:
        cv2.putText(frame, "JUMP!", (cx - 30, cy + hh + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 3)

    # ── Active Keys debug (top-left) ──
    cv2.putText(frame, "Press K or End to quit", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    keys_list = sorted(pressed_keys) if pressed_keys else ["(none)"]
    keys_str = "ACTIVE KEYS: " + " + ".join(k.upper() for k in keys_list)
    cv2.putText(frame, keys_str, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    # ── Debug: hand angle + hotbar (left side) ──
    angle_str = f"Angle: {debug_hand_angle:+.1f} deg"
    slot_str = f"Hotbar: [{hotbar_slot + 1}]"
    cv2.putText(frame, angle_str, (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 50), 1)
    cv2.putText(frame, slot_str, (10, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    # ── Smoothing indicator ──
    buf_status = f"Smooth: {len(gesture_buffer)}/{SMOOTH_FRAMES}"
    cv2.putText(frame, buf_status, (10, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── Second-hand mouse indicator ──
    m2_label = "Mouse2: ACTIVE" if debug_mouse2_active else "Mouse2: (no 2nd hand)"
    m2_clr = (180, 180, 255) if debug_mouse2_active else (100, 100, 100)
    cv2.putText(frame, m2_label, (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, m2_clr, 1)

    # ── Inventory lock banner ──
    if inventory_locked:
        banner = "!! INVENTORY LOCKED - fist to close !!"
        bs = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        bx = (w - bs[0]) // 2
        cv2.putText(frame, banner, (bx, h // 2 + hh + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # ── Finger state dots (bottom-centre) ──
    if finger_states is not None:
        start_x = (w - len(FINGER_NAMES) * 70) // 2
        y_pos = h - 50
        for i, (name, is_up) in enumerate(zip(FINGER_NAMES, finger_states)):
            fx = start_x + i * 70
            dot_clr = (0, 255, 0) if is_up else (0, 0, 200)
            cv2.circle(frame, (fx + 15, y_pos), 8, dot_clr, -1)
            cv2.circle(frame, (fx + 15, y_pos), 8, (255, 255, 255), 1)
            cv2.putText(frame, name, (fx - 2, y_pos + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    # ── Gesture reference panel (right side) ──
    px_start = w - 220
    py_start = 120
    overlay_img = frame.copy()
    cv2.rectangle(overlay_img, (px_start - 5, py_start - 15),
                  (w - 5, py_start + len(GESTURE_REFERENCE) * 22 + 5),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay_img, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "GESTURE REF", (px_start, py_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    for i, (name, desc, clr) in enumerate(GESTURE_REFERENCE):
        y = py_start + 18 + i * 22
        active = any(name.startswith(a.upper()[:5]) for a in actions)
        if name == "JUMP" and jumped:
            active = True
        dot_clr = clr if active else (100, 100, 100)
        cv2.circle(frame, (px_start + 5, y - 4), 4 if active else 3, dot_clr, -1)
        txt_clr = clr if active else (140, 140, 140)
        cv2.putText(frame, f"{name}: {desc}", (px_start + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, txt_clr, 1)

    # ── FPS (bottom-left) ──
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    global running, inventory_locked, mouse2_prev_x, mouse2_prev_y, debug_mouse2_active, last_jump_time

    print("=" * 55)
    print("  Minecraft Gesture Control System  (v3.3)")
    print("=" * 55)
    print("  Kill switch : Press 'K' or 'End' key")
    print("  Mouse: Right hand palm (delta-based)")
    print("  Jump:  3 fingers up (index+mid+ring)")
    print("  Hotbar: thumbs-up + tilt")
    print("  Run as Administrator for DirectInput!")
    print("=" * 55)

    start_kill_switch()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,            # Two hands: gestures + mouse
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    print("[INFO] Camera opened. Starting gesture recognition...")

    prev_actions: set[str] = set()
    last_oneshot_time = 0.0
    no_hand_counter = 0
    prev_time = time.time()

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            actions: set[str] = set()
            outside_dz = False
            finger_states = None
            jumped = False

            if results.multi_hand_landmarks and results.multi_handedness:
                no_hand_counter = 0

                # ── Sort hands by label: Left = gestures, Right = mouse ──
                gesture_hl = None
                gesture_label = None
                mouse_hl = None
                mouse_label = None

                for i, (hl_i, hd_i) in enumerate(zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                )):
                    hand_label = hd_i.classification[0].label
                    # NOTE: MediaPipe mirrors labels on front camera,
                    # so "Left" in MediaPipe = user's left hand on screen.
                    if hand_label == "Left" and gesture_hl is None:
                        gesture_hl = hl_i
                        gesture_label = hand_label
                    elif hand_label == "Right" and mouse_hl is None:
                        mouse_hl = hl_i
                        mouse_label = hand_label

                # ── Process gesture hand (Left) ──
                if gesture_hl is not None:
                    lms = gesture_hl.landmark

                    # Draw skeleton + bounding box for gesture hand
                    mp_drawing.draw_landmarks(
                        frame, gesture_hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    draw_hand_bbox(frame, gesture_hl, fw, fh)

                    # Label on screen
                    gx = int(lms[0].x * fw)
                    gy = int(lms[0].y * fh) + 25
                    cv2.putText(frame, "GESTURE", (gx - 30, gy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Classify raw gesture
                    raw_actions, finger_states = classify_gestures(lms, gesture_label)

                    # Smooth gesture (3-frame buffer)
                    actions = smooth_gesture(raw_actions)

                    # Hotbar scroll (only when in hotbar mode)
                    if "hotbar" in actions:
                        handle_hotbar_scroll(lms)
                    else:
                        compute_hand_angle(lms)



                # ── Process mouse hand (Right) ──
                debug_mouse2_active = False
                if mouse_hl is not None:
                    lms2 = mouse_hl.landmark

                    # Draw second hand skeleton (different colour)
                    mp_drawing.draw_landmarks(
                        frame, mouse_hl, mp_hands.HAND_CONNECTIONS,
                    )

                    # Label on screen
                    mx = int(lms2[0].x * fw)
                    my = int(lms2[0].y * fh) + 25
                    cv2.putText(frame, "MOUSE", (mx - 25, my),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 2)

                    handle_mouse2(lms2, mouse_label, fw, fh)
                else:
                    # Reset tracking when mouse hand disappears
                    mouse2_prev_x = None
                    mouse2_prev_y = None

                # Execute inputs
                last_oneshot_time = execute_actions(
                    actions, prev_actions, last_oneshot_time
                )
            else:
                no_hand_counter += 1
                if no_hand_counter >= NO_HAND_TIMEOUT_FRAMES:
                    if pressed_keys:
                        release_all()
                    inventory_locked = False
                    gesture_buffer.clear()

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 0.001)
            prev_time = now

            # Overlay
            draw_overlay(frame, actions, outside_dz, fps, finger_states, jumped)

            prev_actions = actions

            cv2.imshow("MC Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                running = False

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        print("[INFO] Shutting down...")
        release_all()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Goodbye!")


if __name__ == "__main__":
    main()
