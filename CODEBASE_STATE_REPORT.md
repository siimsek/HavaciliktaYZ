==================================================
PROJECT STATE REPORT
Generated At: 2026-02-22 18:25:37 +03
==================================================

1) Project Identity
- Project name (detectable): HavaciliktaYapayZeka (runtime module: `AIA`, training module: `AIA-training`).
- Primary purpose: TEKNOFEST Havacilikta Yapay Zeka competition pipeline for (a) per-frame object detection + landing suitability output and (b) GPS-loss position estimation.
- Competition/task alignment: Code targets TEKNOFEST Task-1 (object detection + landing status) and Task-2 (position estimation), with local simulation and server-loop execution paths.
- Main problem it solves: Real-time frame-by-frame inference and result submission under offline competition constraints.

2) Repository Overview
- Directory structure summary:
  - `AIA/`: runtime/inference and competition loop.
  - `AIA-training/`: model/data preparation and training pipelines.
  - Root: orchestration-level metadata (`CHANGELOGS.md`) and now this state report.
- Core modules (`AIA/src`):
  - `detection.py`: YOLO inference, preprocessing, post-filtering, landing suitability decision.
  - `movement.py`: lightweight vehicle movement status tracking.
  - `localization.py`: hybrid GPS + optical-flow position estimation.
  - `network.py`: frame retrieval, image download, result POST, retries, payload sanitation.
  - `data_loader.py`: VisDrone-based local simulation iterator.
  - `utils.py`: logging, debug visualization, JSON logging.
- Entry points:
  - `AIA/main.py`: main runtime launcher (interactive menu).
  - `AIA-training/uav_training/train.py`: UAV detection training.
  - `AIA-training/gps_training/train.py`: Siamese GPS-delta training.
- Config files:
  - `AIA/config/settings.py`: central runtime configuration.
  - `AIA/config/task3_params.yaml`: task-3 thresholds (currently not consumed by runtime code).
  - `AIA-training/uav_training/config.py` and `AIA-training/gps_training/config.py`: training configs.
- Model files:
  - Runtime expects `AIA/models/yolov8m.pt`.
  - Training scripts default to Ultralytics checkpoints (`yolo11m.pt`) and save artifacts under `AIA-training/artifacts/...`.
- Utilities:
  - Data audit/build scripts (`AIA-training/uav_training/audit.py`, `build_dataset.py`).
  - GPS dataset audit (`AIA-training/gps_training/audit_gps.py`).

3) System Architecture (Current Implementation)
- Actual implemented architecture:
  - Monolithic orchestrator (`main.py`) invoking modular components for network, detection, movement, localization, visualization.
  - Synchronous pull-process-push loop in competition mode.
  - Separate offline training stack not coupled at runtime.
- Data flow from input to output:
  1. `NetworkManager.get_frame()` pulls frame metadata.
  2. `NetworkManager.download_image()` downloads/decode frame.
  3. `ObjectDetector.detect()` runs preprocessing, inference (standard or SAHI), filtering, landing logic.
  4. `MovementEstimator.annotate()` adds `movement_status` for vehicle detections.
  5. `VisualOdometry.update()` computes current translation estimate.
  6. `NetworkManager.send_result()` emits sanitized JSON payload.
- Detection pipeline:
  - CLAHE + unsharp preprocessing.
  - YOLO predict on full frame and optional tiled SAHI inference.
  - Class-wise custom NMS + containment suppression.
  - Min/max bbox and aspect-ratio filters.
  - COCO-to-competition class remap.
- Tracking/motion logic:
  - Center-point nearest-neighbor matching with distance gating.
  - Fixed history window, missed-frame aging, displacement threshold for movement state.
- Landing suitability logic:
  - For UAP/UAI classes only: edge-touch rejection + obstacle overlap check (`intersection/landing_area`).
  - Vehicles/humans set `landing_status = -1`.
- Position estimation logic:
  - GPS healthy: trust telemetry (`translation_x/y/z`) directly.
  - GPS unhealthy: Lucas-Kanade optical flow over Shi-Tomasi features; median pixel displacement -> meter conversion via focal length and altitude.
- JSON output generation:
  - Sends `frame`, `detected_objects` (without confidence), `detected_translations` list.
  - Includes `movement_status` in each object.
- Server communication flow:
  - Session check (`GET base_url`) -> repeated `GET /next_frame` + image download -> `POST /submit_result`.
  - Retry with timeout and fixed delay.

4) Implemented Features (What Actually Exists)
- Object detection:
  - Implemented: YOLO inference, SAHI slicing option, post-filters, class remapping.
  - Critical gap: current mapping logic cannot produce class `2`/`3` (UAP/UAI) from model outputs as coded.
- Motion classification:
  - Implemented for class `0` only via track history displacement.
  - No temporal confidence model; binary threshold-based status.
- Landing logic:
  - Implemented algorithmically.
  - Functionally blocked if UAP/UAI classes are never emitted.
- Position estimation:
  - Implemented hybrid GPS + optical flow.
  - No camera intrinsics calibration loader; uses static focal fallback.
- Tracking:
  - Implemented lightweight tracker for movement labeling only.
  - No persistent object IDs in output payload.
- Error handling:
  - Broad try/except around module init and frame loop.
  - Network and detection failures generally fail-soft.
- Logging:
  - Implemented console logger with levels and sampled JSON disk logging.
- Config management:
  - Runtime centralized in `settings.py`.
  - Multiple hardcoded constants still present in modules (e.g., edge margin, ratio thresholds).
- Determinism control:
  - Partially implemented; no global seed policy in runtime.
- Offline safeguards:
  - Runtime model path is local-only and no cloud API calls are used.
  - Network layer still allows arbitrary `BASE_URL`; no explicit enforcement that target is local/offline subnet.

Implementation status summary:
- Fully implemented: synchronous runtime loop, network retries, core inference loop, optical-flow fallback, payload emission.
- Partially implemented: movement classification robustness, determinism controls, compliance hardening.
- Missing or effectively non-functional: reliable UAP/UAI production path in current class mapping logic.

5) Determinism & Reproducibility Status
- Seed usage:
  - Runtime: no fixed seeds for Python/NumPy/Torch.
  - DatasetLoader uses random sequence/image selection without deterministic seed.
- Random components:
  - Random selection in simulation datasets.
  - Training data sampling/oversampling paths use random operations.
- Multithreading risks:
  - Runtime largely single-threaded, low race risk.
  - Training dataloaders use multiple workers; order and timing can vary.
- GPU nondeterminism risks:
  - Runtime uses CUDA + FP16; potential non-deterministic kernels.
  - GPS training explicitly enables `torch.backends.cudnn.benchmark = True`.
  - UAV training config sets `deterministic=False`.
- Order sensitivity risks:
  - Tracking assignment and NMS depend on sorted candidate order and floating comparisons.
- Floating point stability risks:
  - FP16 inference and mixed precision training increase minor run-to-run drift.
  - Repeated position integration in optical flow accumulates numeric error.

Assessment: determinism is not controlled to competition-audit-grade standards.

6) Performance Characteristics (Based on Code)
- Frame processing structure:
  - Strict serial chain (fetch -> decode -> detect -> movement -> localization -> submit).
- Per-frame computation flow:
  - High cost at detection stage, especially with SAHI tiled inference.
- Memory handling approach:
  - Periodic `torch.cuda.empty_cache()` in runtime.
  - Simulation image cache and optional frame cache in GPS dataset training.
- Potential bottlenecks:
  - SAHI nested tile loop with per-tile model invocation.
  - Synchronous HTTP/image operations block compute.
  - Debug visualization and disk writes can degrade throughput.
- Blocking operations:
  - All network I/O, image decode, and POST are blocking.
- I/O strategy:
  - In-memory decode path is efficient.
  - Sampled JSON logging mitigates but does not eliminate file I/O overhead.
- Scalability risks:
  - Architecture scales vertically only; no batching, pipelining, or async stages.
  - 4K inputs + SAHI + high detection limits can collapse frame rate on weaker hardware.

7) Competition Compliance Check
- Class ID mapping correctness:
  - Partial/incorrect at runtime level.
  - Vehicle/human mapping exists, but UAP/UAI output path is not represented in the active mapping table; this is a competition-critical defect.
- Landing suitability rule correctness:
  - Logic aligns with rule intent (edge exclusion + any overlap invalidates suitability).
  - Effective only when UAP/UAI detections exist.
- Motion classification logic correctness:
  - Implemented for vehicles only; consistent with payload extension but rule source for this field is not clearly bound in provided spec excerpt.
- JSON format compliance:
  - Mostly compliant with expected structure (`frame`, object array, translation list).
  - Optional fields (`id`, `user`) from spec excerpt are not sent.
- Offline compliance (internet usage detection):
  - No explicit external internet calls in runtime code.
  - No built-in guard preventing accidental public endpoints in `BASE_URL`.
- Hardcoded assumptions:
  - Static focal length, default altitude fallback, fixed edge margin, and fixed thresholds.
  - Assumed telemetry key names (`translation_*`, `gps_health`).

8) Code Quality Assessment
- Modularity:
  - Good module split in runtime (`detection`, `network`, `localization`, `movement`, `utils`).
- Separation of concerns:
  - Generally good, but `main.py` is large and contains orchestration + UX/menu + KPI formatting.
- Config centralization:
  - Strong central config usage in runtime.
  - Some constants remain embedded in methods.
- Magic numbers:
  - Present in multiple places (e.g., edge margin 5, min tracked points 10/5, aspect ratio 6.0).
- Dependency hygiene:
  - Reasonable direct dependencies; no lockfile pinning for strict reproducibility.
- Readability:
  - High inline documentation and clear method naming.
- Technical debt zones:
  - Runtime/docs mismatch (README shows argparse usage; code uses interactive menu).
  - Dead configuration (`task3_params.yaml`) not integrated.
  - Training and runtime conventions diverge (YOLO versions, naming, determinism posture).

9) Risk Zones
- Architectural risks:
  - Single-thread serial pipeline vulnerable to latency spikes.
  - Heavy dependence on one detector path for all downstream decisions.
- Runtime risks:
  - Broad exception handling can mask recurring data-quality failures.
  - Potential silent degradation when detections return empty list.
- Performance risks:
  - SAHI on high-resolution frames may exceed practical per-frame budget.
  - Debug I/O can block real-time behavior if enabled during competition.
- Determinism risks:
  - No seeded runtime, mixed precision, and non-deterministic CUDA settings.
- Competition failure risks:
  - Inability to emit UAP/UAI classes from current mapping logic.
  - Possible mismatch with required payload fields depending on final server contract.

10) Missing Components
- To be production-ready:
  - Structured health metrics/export (latency percentiles, failure categories).
  - Strong validation contracts and schema checks for inbound/outbound JSON.
  - Automated tests for core logic (class mapping, landing decisions, odometry transitions).
  - Asynchronous/pipelined architecture for predictable throughput.
- To be competition-ready:
  - Correct and verified UAP/UAI detection/class mapping path.
  - End-to-end contract tests against official competition server emulator.
  - Deterministic run profile and controlled fallback behaviors.
  - Explicit offline guardrails (endpoint allowlist/local subnet enforcement).
- To be research-grade:
  - Reproducible experiment management (fixed seeds, env capture, versioned datasets).
  - Quantitative benchmarking suite (accuracy, drift, robustness under perturbations).
  - Ablation studies for SAHI, preprocessing, and motion thresholds.

11) Technical Maturity Score
- Architecture maturity: 6/10
- Determinism safety: 3/10
- Performance optimization: 6/10
- Competition robustness: 4/10
- Maintainability: 7/10

12) Summary Verdict
- Stability: Moderately stable under expected happy-path runtime conditions.
- Fragility: Fragile under compliance-critical scenarios and edge-case data quality.
- Competition readiness: Not competition-ready in current state.
- Biggest weakness: Competition-critical class output alignment (UAP/UAI path) is not operationally guaranteed by implemented mapping logic; this undermines landing-status functionality and scoring potential.
