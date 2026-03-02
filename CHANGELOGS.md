# CHANGELOGS

## 0.0.34 - 2026-03-01
- **fix(network)**: Fixed E999 SyntaxError — `if response.status_code == 204` and 5xx branches moved inside `try` block with correct indentation.
- **chore(lint)**: Resolved flake8 errors project-wide (W291/W293 trailing/blank whitespace, E261/E302/E303/E305, E501 line length, F401 unused imports, E731 lambda, W391).
- **chore(config)**: Removed trailing whitespace in settings.py, fixed inline comment spacing.
- **chore(main)**: Added noqa for path-dependent imports, fixed formatting and blank-line conventions.
- **chore(src)**: Cleaned frame_context, detection, localization, movement; removed unused imports.
- **chore(tests)**: Removed unused imports, fixed E303/E501/E731; replaced lambdas with `_summary_cb` helper.
- **chore(tools)**: Fixed mock_server E302, W293.

## 0.0.33 - 2026-03-01
- **refactor(data_loader)**: Recursive scan of datasets/; find images by extension (.jpg, .png, etc.); removed sequences/images folder structure requirement.
- **refactor(config)**: Removed DATASET_NAME_PATTERN, added IMAGE_EXTENSIONS; supports any dataset structure.
- **chore(comments)**: Cleaned up redundant comments, added concise explanatory comments at critical points.
- **docs(data_loader)**: Replaced VisDrone references with generic wording.

## 0.0.32 - 2026-03-01
- **fix(audit)**: Applied fixes per TEKNOFEST audit reports.
- **fix(main)**: Use odometry.get_position() instead of (0,0,0) fallback on image download failure (B-04).
- **fix(movement)**: Vehicle selection supports both cls_int and cls (str/int) comparison (KRITIK-01).
- **fix(localization)**: Added log warning when FOCAL_LENGTH_PX=800 default is used (B-13).
- **fix(config)**: Optional task3_params.yaml loading; Task 3 params loaded from YAML when present (KRITIK-03).
- **fix(detection)**: Removed unreachable dead code from _is_touching_edge_raw.
- **chore(deps)**: Added pytest>=7.0.0, pytest-timeout>=2.0.0, PyYAML>=6.0 to requirements.txt.
- **docs(readme)**: Updated audit table, setup, configuration, and file structure.
- **chore(comments)**: Cleaned up obsolete audit reference comments.

## 0.0.31 - 2026-03-01
- **fix(typing)**: Added `TYPE_CHECKING` imports for `FrameContext` in `src/localization.py` and `src/movement.py` to fix Flake8 `F821` syntax errors during CI build without introducing circular dependencies.
- **release**: Bumped project version from `0.0.30` to `0.0.31`.

## 0.0.30 - 2026-03-01
- **fix(tests)**: Updated `MovementEstimator.annotate` mock and test calls to use `frame_ctx` instead of `frame`.
- **fix(tests)**: Updated `TestRiderSuppression` overlapping coordinates to correctly test the new comprehensive vehicle suppression logic.
- **fix(main)**: Fixed `NameError` crash in competition loop exception handling by explicitly capturing `pending_result_snapshot`.
- **fix(main)**: Prevented `run_competition` from indefinitely swallowing exceptions and hanging when running under `pytest`.
- **chore(tests)**: Deleted duplicate and obsolete mock-based test runner (`run_mocked_tests.py`) in favor of centralized robust `pytest` suite.
- **release**: Bumped project version from `0.0.29` to `0.0.30`.

## 0.0.29 - 2026-03-01
- **fix(movement)**: Hardened `_age_tracks` dict iteration with `list()` wrapper to prevent `RuntimeError` on future mutation (§2.2).
- **fix(movement)**: Added explicit `None` guard after `goodFeaturesToTrack` fallback in `_estimate_camera_shift` to prevent `TypeError` crash (§2.3).
- **fix(network)**: Changed JSON `ValueError` handler from immediate `FATAL_ERROR` to transient retry with backoff (§2.5).
- **fix(network)**: Added session timestamp to idempotency key format (`prefix:session_id:frame_id`) to prevent cross-session key collisions (§6).
- **fix(detection)**: Added `np.maximum(areas, 1e-6)` guard to `_nms_greedy` to prevent zero-area bbox math issues (§9).
- **fix(main)**: Wrapped `cv2.imshow`/`cv2.waitKey` in `try/except cv2.error` for headless environment safety (§7).
- **chore(deps)**: Replaced `opencv-python` with `opencv-python-headless` in `requirements.txt` to avoid GUI dependency crashes (§4/§7).
- **test(tests)**: Updated idempotency key test assertion to match new session-scoped key format.

## 0.0.28 - 2026-02-28
- **fix(localization)**: Added EMA smoothing (α=0.4), last-GPS-altitude fallback, and max displacement clamping to optical flow for drift resistance.
- **fix(detection)**: Hardened exception handling — OOM isolated, SystemExit/KeyboardInterrupt re-raised, periodic `empty_cache()` anti-pattern removed.
- **fix(detection)**: Added `kind="stable"` to NMS and containment suppression `np.argsort` for run-to-run reproducibility.
- **fix(movement)**: Clamped `_cam_total_x/y` to ±1e6 to prevent floating-point precision loss over long sessions.
- **fix(data_loader)**: Replaced deterministic GPS degradation cycle with 33% random probability after frame 450.
- **fix(image_matcher)**: Added degenerate/collinear point guard before `cv2.findHomography` to prevent RANSAC crashes.
- **chore(config)**: Marked `task3_params.yaml` as deprecated dead code; all values hardcoded in `settings.py`.
- **test(tests)**: Consolidated 13 test files into single `tests/test_all.py` (47 tests, ~6s); added 10s global timeout via `conftest.py`.
- **docs(readme)**: Added Audit & Hardening section, updated file structure, features table, Task 3 deprecation notice.

## 0.0.27 - 2026-02-26
- **feat(task3)**: Added `src/image_matcher.py` (ORB/SIFT feature matching + homography) and integrated it into main competition/simulation loops.
- **fix(detection)**: Safely bypassed `MAX_BBOX_SIZE` filter by increasing it to `9999` to ensure large vehicles like buses and trains are not filtered.
- **feat(network)**: Wired `detected_undefined_objects` throughout the payload builder and preflight validation layers.
- **fix(simulation)**: Corrected robust GPS health simulation in `data_loader.py` to match the exact requirements of `sartname.md` (first 450 frames healthy, switching to cyclic degradation).
- **feat(testing)**: Created `tools/mock_server.py` to provide a complete standalone local testing environment perfectly simulating the TEKNOFEST competition HTTP format.
- **docs(readme)**: Completely realigned `README.md` with the latest `sartname.md` details (updated table of contents, three-task description, extended technical constraint matrices, architecture diagrams).

## 0.0.26 - 2026-02-26
- Added outbound payload preflight validation in `src/network.py` to enforce required top-level fields, strict translation schema (`len==1`), per-object normalization, and safe-fallback conversion on invalid payloads.
- Implemented deterministic object-count protection with class quota + global cap (`RESULT_CLASS_QUOTA`, `RESULT_MAX_OBJECTS`) and structured clip telemetry logging.
- Reworked `send_result` to explicit `SendResultStatus` flow and added 4xx handling that sends a single safe-fallback payload before classifying permanent rejection.
- Updated competition send-state handling in `main.py` to branch on status outcomes and publish new KPI counters for fallback ACK, permanent reject, preflight reject, and payload clipping.
- Added tests: `tests/test_network_payload_guard.py` and `tests/test_main_ack_state_machine.py`.

## 0.0.25 - 2026-02-25
- Implemented adaptive HTTP timeout split in `src/network.py` with separate connect/read budgets for frame metadata, image download, and result submission (`REQUEST_CONNECT_TIMEOUT_SEC`, `REQUEST_READ_TIMEOUT_SEC_*`) while preserving fallback compatibility with legacy `REQUEST_TIMEOUT`.
- Added jittered exponential backoff (`BACKOFF_BASE_SEC`, `BACKOFF_MAX_SEC`, `BACKOFF_JITTER_RATIO`) and wired it into network retry paths to reduce chained timeout stalls under transient network turbulence.
- Added duplicate frame/client idempotency safeguards in `src/network.py`: seen-frame LRU (`SEEN_FRAME_LRU_SIZE`), duplicate marking via `FrameFetchResult.is_duplicate`, `Idempotency-Key` header generation (`IDEMPOTENCY_KEY_PREFIX`), and strict duplicate submit short-circuit after successful ACK.
- Updated `main.py` competition runtime to drop duplicate frames before expensive inference/submit stages and extended KPI telemetry with `frame_duplicate_drop` + timeout counters (`timeout_fetch`, `timeout_image`, `timeout_submit`).
- Added dependency-aware test coverage for hardening changes in `tests/test_network_timeouts.py`, `tests/test_frame_dedup.py`, `tests/test_idempotency_submit.py`, and `tests/test_competition_loop_hardening.py`.

## 0.0.24 - 2026-02-25
- Enforced competition-safe FP32 determinism in `main.py`: runtime now force-applies `deterministic-profile=max` in `--mode competition`, even when CLI requested `off` or `balanced`.
- Added explicit override warning telemetry in `main.py` to reduce operator misconfiguration risk on final runs.
- Extended runtime profile observability in `src/runtime_profile.py` to log both `requested` and `effective` profile values for auditability.
- Updated deterministic profile guidance in `README.md`: competition path requires `max` (FP32), while `balanced` remains the speed-oriented simulation profile.

## 0.0.23 - 2026-02-25
- Added wall-clock aware session resilience layer (`src/resilience.py`) with `NORMAL/DEGRADED/OPEN` breaker states, transient event windows, cooldown-based half-open transition, and soft-first abort policy.
- Refactored `main.py` competition loop to use resilience orchestration: transient/ACK storms trigger degrade/open flow instead of immediate budget abort, while `FATAL_ERROR` and `204 END_OF_STREAM` behavior remains strict.
- Implemented fetch-only degrade mode in competition flow with configurable heavy-pass interval (`DEGRADE_SEND_INTERVAL_FRAMES`) and stale pending TTL protection for degraded frames.
- Extended `NetworkManager.send_result(..., degrade: bool = False)` logging for resilience telemetry without changing payload schema semantics.
- Added new resilience configuration knobs in `config/settings.py` (`CB_TRANSIENT_WINDOW_SEC`, `CB_TRANSIENT_MAX_EVENTS`, `CB_OPEN_COOLDOWN_SEC`, `CB_MAX_OPEN_CYCLES`, `CB_SESSION_MAX_TRANSIENT_SEC`, `DEGRADE_FETCH_ONLY_ENABLED`, `DEGRADE_SEND_INTERVAL_FRAMES`).
- Added unit tests in `tests/test_session_resilience.py` covering transient window trigger, OPEN→DEGRADED→NORMAL recovery path, wall-clock abort, and breaker open-cycle abort.
- Updated `README.md` determinism section to clarify that wall-clock is used only for network resilience orchestration, not model decision logic.

## 0.0.22 - 2026-02-23
- Implemented camera-motion compensation in `src/movement.py` using global median optical flow and cumulative camera-shift subtraction for `motion_status` stability under panning motion.
- Extended `MovementEstimator.annotate` signature to `annotate(detections, frame=None)` with backward-compatible fallback when frame is omitted.
- Integrated motion compensation into both simulation and competition loops by passing `frame` from `main.py`.
- Added rider suppression in `src/detection.py` to remove `person` detections overlapping with two-wheeler vehicles (bicycle/motorcycle/motor) based on overlap or IoU thresholds.
- Preserved source class lineage via `source_cls_id` in detection parsing to support two-wheeler-specific suppression decisions.
- Added configuration controls in `config/settings.py` for motion compensation and rider suppression thresholds/flags.
- Updated `README.md` with new configuration tables, motion-comp behavior, and scooter approximation note.
- Added lightweight unit tests in `tests/test_movement_compensation.py` and `tests/test_rider_suppression.py` covering planned motion and rider suppression scenarios with dependency-aware skips.
- Hardened `main.py` competition loop with a "no frame left behind" state machine: when image download fails, the runtime now sends a mandatory fallback result (`detected_objects=[]`, zero translations) instead of skipping the frame.
- Enforced ACK-gated progression: the runtime no longer fetches a new frame while a previous frame result is unacknowledged, and retries result submission with bounded backoff plus failure budget for protocol-safe behavior.
- Added a debug safety guard so visualizer drawing is skipped cleanly on fallback frames where no decoded image exists.

## 0.0.21 - 2026-02-23
- Aligned frame metadata handling with TEKNOFEST technical-spec field names in `src/network.py` by normalizing `frame_id` from `frame_id/id/url/frame`, `frame_url` from `image_url`, and `gps_health` from `gps_health_status`.
- Updated simulation frame metadata to include specification-style keys (`id`, `url`, `image_url`, `session`, `gps_health_status`) while preserving existing internal compatibility keys.
- Refreshed `CODEBASE_STATE_REPORT.md` generation timestamp to match the latest codebase scan.
- Fixed stale compliance wording in state report: motion classification is now documented as outbound via per-object `motion_status`.
- Updated state report test/risk posture to reflect current reality: existing motion-compensation unit tests are acknowledged while broader integration coverage remains pending.

## 0.0.20 - 2026-02-23
- Closed `movement_status` vs `motion_status` integration risk by dual-writing motion labels in `src/movement.py` and accepting both keys in `src/network.py` payload serialization.
- Kept outbound API field name as `motion_status` while preserving backward compatibility for internal consumers still using `movement_status`.

## 0.0.19 - 2026-02-23
- Synced `CODEBASE_STATE_REPORT.md` with the current outbound JSON contract so documentation now matches runtime behavior (`id`, `user`, `frame`, per-object `motion_status`, and `detected_undefined_objects`).
- Removed stale strict-minimal payload statements from the live state report to keep codebase-wide JSON schema references consistent.

## 0.0.18 - 2026-02-23
- Updated outbound result payload in `src/network.py` from strict-minimal to a specification-aligned schema including top-level `id`, `user`, `frame`, `detected_objects`, `detected_translations`, and `detected_undefined_objects`.
- Added per-object `motion_status` field (mapped from internal `movement_status` with safe fallback `-1`) to align Task-1 movement reporting with specification terminology.
- Passed live frame metadata from `main.py` to payload builder so `id/user/frame` can be populated from server-provided fields when available.
- Updated `README.md` JSON example and notes to reflect the expanded payload schema.

## 0.0.17 - 2026-02-23
- Removed `AIA/sartname/UYUMLULUK_RAPORU_2026-02-23.md` on user request and switched compliance reporting to inline chat delivery.

## 0.0.16 - 2026-02-23
- Added a formal specification compliance report at `AIA/sartname/UYUMLULUK_RAPORU_2026-02-23.md` by cross-checking `teknofest_context.md` and TEKNOFEST 2026 technical specification PDF (V1.0, 21.02.2026).
- Documented itemized compliance status (Compliant/Partial/Non-compliant), prioritized risks (P0/P1/P2), and concrete closure actions for Task-3, payload schema alignment, motion status naming, and offline policy guard.

## 0.0.15 - 2026-02-22
- Refreshed `CODEBASE_STATE_REPORT.md` to reflect latest runtime architecture updates, including determinism profiles, fetch-state based networking, strict-minimal payload behavior, pinned dependencies, and revised technical maturity/risk assessment.

## 0.0.14 - 2026-02-22
- Hardened competition networking with explicit frame fetch states (`OK`, `END_OF_STREAM`, `TRANSIENT_ERROR`, `FATAL_ERROR`) and updated loop behavior to avoid premature session termination on transient failures.
- Fixed result submission retry behavior: non-200 responses no longer short-circuit retries; failures now retry up to `MAX_RETRIES` with clearer diagnostics.
- Added strict-minimal payload builder in `src/network.py` and removed `movement_status` from outbound competition JSON; added per-frame bbox normalization/clamp and field/type validation before send.
- Added deterministic runtime profiles via `src/runtime_profile.py` and wired startup bootstrap in `main.py` (`--deterministic-profile off|balanced|max`), with balanced defaults disabling TTA while keeping FP16 enabled.
- Refactored startup flow to CLI-first execution (`--mode competition|simulate_vid|simulate_det`) with optional legacy menu via `--interactive`.
- Strengthened JSON logging integrity in `src/utils.py` with safe filename sanitization, log retention pruning, and visible warning logs for write failures.
- Pinned runtime dependencies in `requirements.txt` to fixed versions for reproducibility.
- Updated README usage and configuration sections to document CLI-first startup, deterministic profiles, and strict payload behavior.

## 0.0.13 - 2026-02-22
- Added `CODEBASE_STATE_REPORT.md` with a structured, timestamped technical audit of current architecture, implementation status, determinism profile, performance characteristics, and competition compliance risks.

## 0.0.12 - 2026-02-22
- Added GitHub Actions workflow at `.github/workflows/ci.yml` to run an automated quality gate on `push` and `pull_request` to `main`.
- CI now performs syntax-focused `flake8` checks (`E9,F63,F7,F82`) to fail fast on parse/name-critical issues.
- Added `python -m compileall -q .` compile pass to detect syntax regressions early without executing runtime code.
- Kept pipeline dependency-light (no heavy ML package install) to reduce CI duration and avoid false negatives from GPU stack setup.

## 0.0.11 - 2026-02-22
- Added end-of-session KPI counters in competition mode summary: `Send OK`, `Send FAIL`, `Mode GPS`, and `Mode OF`.
- Wired competition loop to accumulate send-status and localization-mode totals without changing existing warning/error behavior.

## 0.0.10 - 2026-02-22
- Added interval-based compact KPI result line in competition mode (`main.py`) for better operator visibility beyond warning/error logs.
- Introduced `COMPETITION_RESULT_LOG_INTERVAL` setting (default: `10`) to control how often frame-level KPI lines are printed.
- KPI output now includes frame id, object count, send status, GPS/OF mode, and x/y/z with safe fallbacks for missing telemetry.

## 0.0.09 - 2026-02-22
- Updated terminal banner year in `main.py` from 2025 to 2026 for consistency with project documentation.

## 0.0.08 - 2026-02-22
- Added explicit memory cleanup in `src/movement.py` by clearing track history before stale track deletion.
- Hardened incoming frame validation in `src/network.py` with defensive type sanitization for `gps_health`, `translation_*`, and `altitude` fields.
- Added safe fallbacks for null/unknown/invalid telemetry values to improve runtime resilience under noisy server payloads.

## 0.0.07 - 2026-02-21
- Added `config/task3_params.yaml` as the explicit Task-3 parameter file for `T_confirm`, `T_fallback`, `N`, and `grid stride`.
- Documented the Task-3 parameter contract and field mapping in `README.md`.

## 0.0.06 - 2026-02-21
- Added a dedicated "Task 1 Temporal Decision Logic" section to `README.md` documenting window, decay, and threshold based decision flow.
- Explicitly prohibited single-frame final decisions for `movement_status` and `landing_status` in documentation.

## 0.0.05 - 2026-02-21
- Added a new "Determinism Contract" section to `README.md` with explicit rules for fixed seeds, model eval mode, version pinning, and stable JSON key ordering.
- Added the section link to README table of contents for easier navigation.

## 0.0.04 - 2026-02-21
- Updated all year references in `README.md` from 2025 to 2026 to align project documentation with TEKNOFEST 2026 scope.

## 0.0.03 - 2026-02-21
- Added sampled JSON logging controls (`ENABLE_JSON_LOGGING`, `JSON_LOG_EVERY_N_FRAMES`) and applied frame/result log sampling in `src/network.py` to reduce disk I/O pressure.
- Removed duplicate image writes in simulation save mode by disabling Visualizer auto-save when per-frame save is already active.

## 0.0.02 - 2026-02-21
- Added `movement_status` support for Task-1 output using a lightweight temporal tracker (`src/movement.py`) based on vehicle centroid history.
- Integrated movement annotation into both simulation and competition loops in `main.py`.
- Extended network payload serialization to include `movement_status` with safe fallback (`-1`) when missing.
- Added movement estimator tuning parameters to `config/settings.py`.

## 0.0.01 - 2026-02-21
- Fixed P1 mode-safety issue: selecting competition mode now forces `NetworkManager` to run with `simulation_mode=False`, even if `Settings.SIMULATION_MODE=True`.
- Refactored network mode checks to use instance-level `self.simulation_mode` instead of global settings so runtime mode is explicit and controllable.

## 0.0.19 - 2026-03-02
- Updated `README.md` terminology from "Deterministiklik" to "Tutarlılık ve Tekrarlanabilirlik (Best-Effort)".
- Softened wording from strict/guarantee language to variance-reduction guidance (best-effort consistency framing).
- Revised profile guidance text to present `max` as a stability-oriented recommendation instead of a hard requirement.

## 0.0.20 - 2026-03-02
- Added runtime environment switching overrides in `main.py` to reduce competition-day config risk without code edits.
- Added `--base-url` and `--team-name` CLI flags.
- Added `AIA_BASE_URL` and `AIA_TEAM_NAME` environment variable support (CLI takes precedence).
- Runtime now logs active override source for traceability.

## 0.0.21 - 2026-03-02
- Added GPS=0 latency compensation flow with monotonic fetch timestamp capture and submit-time projection in `main.py`, without changing outbound payload schema.
- Added EMA-based velocity/projection helper in `src/localization.py` and wired it to position updates for dynamic runtime compensation.
- Added feature/config controls: `LATENCY_COMP_ENABLED`, `LATENCY_COMP_MAX_MS`, `LATENCY_COMP_MAX_DELTA_M`, `LATENCY_COMP_EMA_ALPHA`.
- Added compensation KPIs (`compensation_apply_count`, `compensation_avg_delta_m`, `compensation_max_delta_m`) to competition summary logs.
- Added unit tests in `tests/test_all.py` for GPS gating, runtime dt usage (non-hardcoded), clamp behavior, and feature-off backward compatibility.
- Updated README configuration section with latency compensation usage notes.

## 0.0.22 - 2026-03-02
- Closed K1 payload contract risk by introducing canonical schema utilities (`src/payload_schema.py`) and enforcing `motion_status` as the only outbound field while still normalizing legacy `movement_status` on input.
- Added startup payload self-check (`NetworkManager.assert_contract_ready`) and switched `MOTION_FIELD_NAME` default to `motion_status` in `config/settings.py`.
- Closed K2 GPS fallback risk by extending `VisualOdometry` with deterministic mode/meta tracking and `predict_without_measurement` flow for GPS unhealthy / missing-measurement paths.
- Updated competition fallback in `main.py` to use predict-only localization updates instead of replaying last pose when frame download fails under `gps_health=0`.
- Closed K3 error-handling risk by adding typed competition error taxonomy + decision policy (`src/competition_contract.py`) and integrating decision-driven runtime handling (`RETRY/DEGRADE/STOP`) in the competition loop.
- Added unit coverage in `tests/test_all.py` for payload schema canonicalization, error policy mapping, and predict-only odometry metadata.
