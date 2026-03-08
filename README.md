# Minecraft Gesture Control System
> Play Minecraft using only your hands — no keyboard required.

A real-time dual-hand gesture control system built with MediaPipe and OpenCV. Your left hand controls all game actions through a hierarchical gesture classifier, while your right hand controls mouse look through delta-based finger tracking. Built and tested on Windows with a standard webcam.

---

## How It Works

The system uses two hands simultaneously with distinct roles:

**Left hand → Game actions** via an 8-level hierarchical gesture classifier with 3-frame smoothing to prevent misfires.

**Right hand → Mouse look** via delta-based index finger tracking with EMA (Exponential Moving Average) smoothing to reduce jitter.

### Gesture Reference

| Gesture | Action | Input |
|---|---|---|
| Open palm (all 5 up) | Inventory | `E` |
| Fist (all curled) | Stop / Close inventory | Release all keys |
| Index + Middle up | Sprint forward | `Ctrl + W` |
| Index + Middle + Ring up | Sprint jump | `Ctrl + W + Space` |
| Thumb-index pinch | Attack | Left click |
| 4 fingers up (thumb down) | Interact | Right click |
| Pinky up (combinable) | Sneak | `Shift` |

### System Architecture

```
Webcam Input
    │
    ▼
MediaPipe Hand Detection (dual hand, 21 landmarks each)
    │
    ├── Left Hand ──► Gesture Classifier (8-level priority hierarchy)
    │                      │
    │                 3-frame smoothing buffer
    │                      │
    │                 pydirectinput (DirectInput keystrokes)
    │
    └── Right Hand ──► Delta tracker (index finger tip)
                           │
                      EMA smoothing (α = 0.4)
                           │
                      pydirectinput (relative mouse move)
```

---

## Technical Details

**Gesture stability** — A 3-frame rolling buffer requires the same gesture to appear in consecutive frames before firing, eliminating single-frame misfires that cause unintended key toggling.

**Pinch hysteresis** — Two separate thresholds for entering (`0.055`) and exiting (`0.08`) pinch state prevent flickering when the hand hovers near the threshold boundary.

**Delta-based mouse** — Rather than mapping absolute finger position to screen position (which causes drift), only the *movement* of the finger between frames is used. A fist resets tracking so there's no position jump when the hand reopens.

**No-hand timeout** — If no hand is detected for 8 consecutive frames, all keys are released automatically to prevent stuck inputs.

---

## Known Limitations

- **Mouse look is a work in progress** — the current delta-based tracking with EMA smoothing works but lacks precision at slow hand speeds. Improving mouse control through more robust finger tracking is an ongoing goal.
- **Windows only** — uses `pydirectinput` for DirectInput, which is Windows-specific. Standard Minecraft Java Edition requires DirectInput to register keystrokes in-game.
- **Must run as Administrator** — required for DirectInput to send inputs to the game window.
- **Tested on one webcam** — should work with any standard webcam at `CAMERA_INDEX = 0`, but lighting conditions significantly affect detection reliability.

---

## Requirements

- Python 3.11
- Windows
- A standard webcam
- Minecraft Java Edition (or any game using standard keyboard/mouse input)

---

## Installation

```bash
git clone https://github.com/prisha7217/Minecraft-CV
cd minecraft-gesture-control
pip install -r requirements.txt
```

---

## Usage

1. Run as Administrator (required for DirectInput)
2. Open Minecraft and load into a world
3. In a separate terminal:

```bash
python gesture_control.py
```

4. Position yourself in front of your webcam with both hands visible
5. Press `K` or `End` at any time to quit safely

---

## Configuration

Key parameters are at the top of `gesture_control.py` and can be tuned without touching the core logic:

| Parameter | Default | Description |
|---|---|---|
| `MOUSE2_SENSITIVITY` | `2.5` | Mouse look speed |
| `MOUSE2_EMA_ALPHA` | `0.4` | Smoothing (lower = smoother) |
| `SMOOTH_FRAMES` | `3` | Gesture stability frames |
| `PINCH_ENTER` | `0.055` | Pinch activation threshold |
| `CAMERA_INDEX` | `0` | Webcam index |

---

## Built With

- [MediaPipe](https://mediapipe.dev/) — hand landmark detection
- [OpenCV](https://opencv.org/) — webcam capture and visual overlay
- [pydirectinput](https://github.com/learncodebygaming/pydirectinput) — DirectInput keystrokes
- [pynput](https://pynput.readthedocs.io/) — kill switch listener

---

## Author

**Prisha Gupta** — [github.com/prisha7217](https://github.com/prisha7217)
