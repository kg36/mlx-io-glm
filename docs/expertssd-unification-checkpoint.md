# Milan ExpertSSD checkpoint — 2026-09-08

This records the native implementation through `baa16015`, paired with Milan
application implementation `368ccf7`. The starting unified native revision was
`b72fe000`. This is a development checkpoint, not a completed migration or release.

## Evidence and acceptance

The application record is `docs/expertssd-unification.md` in `kg36/livseek`.
Its task-local artifact directory is `artifacts/expertssd-unification/`.

- Combined tests: 1,171 passed (`common-prefill-plan-suite.log`).
- Matched DeepSeek target and Preview median throughput differences were -1.23%
  and -1.65%, respectively. Run evidence is in `prefill-ds-abba/` and
  `prefill-preview-repeat-abba/`.
- GLM captured target/MTP smokes matched tokens, routes and FP32 logits. These
  captures are correctness evidence, not throughput qualification.
- The user confirmed that both models were within 2% of remembered best controls
  and accepted this difference as noise. This does not replace final matched
  qualification or weaken correctness and memory checks.

The tested development wheel has SHA-256
`bbb90e74c3453c41279c344a30c754775d717b72745cbd1c80b862e2cc6a343b`.
Its version contains `57c5403c` plus the lease changes committed as `baa16015`.
A clean-revision rebuild is still needed before publishing or updating the pin.

## Remaining work

DeepSeek public target/drafter wiring, legacy snapshot import, startup override
resolution, common prefill telemetry, GLM profile qualification, duplicate-path
removal and final sustained AB/BA qualification remain open. The GLM shared-engine
candidate remains opt-in. The application serving pin and overlay are unchanged.

This checkpoint adds documentation only. It does not change runtime behavior,
model weights, allocation defaults or performance settings.
