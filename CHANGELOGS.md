# CHANGELOGS

## 0.0.01 - 2026-02-21
- Fixed P1 mode-safety issue: selecting competition mode now forces `NetworkManager` to run with `simulation_mode=False`, even if `Settings.SIMULATION_MODE=True`.
- Refactored network mode checks to use instance-level `self.simulation_mode` instead of global settings so runtime mode is explicit and controllable.

## 0.0.02 - 2026-02-21
- Added `movement_status` support for Task-1 output using a lightweight temporal tracker (`src/movement.py`) based on vehicle centroid history.
- Integrated movement annotation into both simulation and competition loops in `main.py`.
- Extended network payload serialization to include `movement_status` with safe fallback (`-1`) when missing.
- Added movement estimator tuning parameters to `config/settings.py`.
