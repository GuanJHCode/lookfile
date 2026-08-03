# SpecStyle

SpecStyle is a StyleOps engine for AMD Radeon GPUs. It compiles a versioned
Style Spec into generation and verification plans, runs a local SDXL pipeline,
routes verified artifacts, and publishes an auditable export bundle.

The implementation keeps three planes separate:

- Generation controls: SDXL img2img, IP-Adapter style conditioning, and Canny
  ControlNet structure conditioning.
- Verification: deterministic L1 checks plus separately versioned L2/L3
  contracts. ControlNet is never treated as a verifier.
- Repair policy: allow-listed rule-to-action decisions, guardrails, bounded
  retries, and full re-verification.

## Current verified scope

The following path has been exercised on an AMD Radeon ROCm 7.2.1 runtime:

1. Load a local-only, hash-verified SDXL base, IP-Adapter Plus SDXL ViT-H, and
   ControlNet Canny supply.
2. Upload one source image, one style reference, and one Style Spec through the
   authenticated Gradio application.
3. Run a production job or a sequential two-to-four variation batch.
4. Observe live persisted job status, request cancellation, and display approved
   and rejected outputs, QA rows, deterministic seed evidence, and the published
   bundle digest.

The current AMD candidate is `f0d258d`. It serves 49 Gradio components on the
isolated candidate port and passed 846 affected AMD tests with one capability
skip, plus repository-configured Ruff check and format gates. The stable
`1a1af45` service remained online during candidate validation.

The Radeon runtime used PyTorch `2.9.1+gitff65f5b` with HIP
`7.2.53211-e1a6bc5663` and Diffusers `0.39.0`. Real GPU evidence was collected
on the following commits, all of which are ancestors of the current candidate:

- `a3832be`: one interactive job traversed `OPENING`, `GENERATING`,
  `VERIFYING`, `CLEANUP`, and `COMPLETED`; controls stayed responsive during
  generation. A separate real job was cancelled during `GENERATING` and
  published no bundle.
- `ec1e27d`: a two-item engineering smoke completed with distinct deterministic
  seeds and artifact hashes. This proves batch wiring, not calibrated
  perceptual diversity.
- `ec1e27d`: two new synthetic source materials reused the same Spec, style
  reference, compiler graph, rules, and environment and both completed. This is
  new-material contract reuse, not same-input semantic replay.

## Known submission gaps

- Preview is a compile-time contract only. No approved Turbo or LCM supply is
  included, and Preview must not be routed through the Production pipeline.
- L2 remains advisory until real held-out human annotations replace the current
  placeholder calibration evidence. No universal L2 threshold is claimed.
- No calibrated L3 domain plugin is enabled. `NOT_APPLICABLE` and
  `UNVERIFIABLE` are not presented as `PASS`.
- A real failure-to-repair-to-reverification Radeon comparison, same-input
  semantic replay evidence, five-arm equal-budget ablation, and blind human
  evaluation are still required for the full competition claim.
- The current production upload contract accepts `xhs_grid`; the other output
  profiles are not yet productized.

## Repository layout

```text
deployment/amd/                 AMD environment probe, lock, and launcher
src/specstyle/generation/       generation contracts and Diffusers adapter
src/specstyle/spec/             schema, loader, compiler, migration, replay
src/specstyle/verification/     L1, L2, L3, routing
src/specstyle/repair/           actions, policy, guardrails, bounded loop
src/specstyle/workflow/         job state, production runtime, export lifecycle
src/specstyle/ui/               injectable Gradio application
tests/                          CPU-safe unit, property, and integration tests
```

## AMD runtime requirements

- Linux with CPython 3.12.
- ROCm 7.2.1 and a working PyTorch HIP installation supplied by the AMD image.
- A Radeon device visible through `torch.cuda` and capable of FP16 execution.
- Local model weights and the matching model descriptors, weight manifests,
  license approvals, and compiler context under the runtime root.

Do not install a generic PyPI `torch`, `torchvision`, or `torchaudio` wheel. The
installer deliberately uses `--no-deps`, rejects those package names, and checks
that the AMD image's Torch binaries did not change.

## Install pinned application dependencies

The checked-in runtime lock is
[`deployment/amd/requirements.lock.txt`](deployment/amd/requirements.lock.txt).
It is synchronized with the guarded installer by a unit test.

```bash
cd /path/to/lookfile
export PYTHONDONTWRITEBYTECODE=1
install -d -m 700 /workspace/persistence/lookfile-runtime/reports/install

/opt/venv/bin/python deployment/amd/scripts/probe_environment.py \
  --phase pre \
  --repo-sha "$(git rev-parse HEAD)" \
  --json-out /workspace/persistence/lookfile-runtime/reports/install/pre.json

/opt/venv/bin/python deployment/amd/scripts/install_dependencies.py \
  --repo-sha "$(git rev-parse HEAD)" \
  --json-out /workspace/persistence/lookfile-runtime/reports/install/install.json

/opt/venv/bin/python deployment/amd/scripts/probe_environment.py \
  --phase post \
  --baseline-json /workspace/persistence/lookfile-runtime/reports/install/pre.json \
  --repo-sha "$(git rev-parse HEAD)" \
  --json-out /workspace/persistence/lookfile-runtime/reports/install/post.json
```

All evidence paths must be canonical absolute paths. Each command exits nonzero
when the host, GPU, dependency, or Torch-integrity contract fails.

## Runtime layout

The launcher expects this layout by default:

```text
/workspace/persistence/lookfile-runtime/
  config/
    context.json
    models.json
    weight_manifests.json
    license_approvals.json
  evidence/
    sha256/<first-two-hex>/<full-sha256>
  models/<manifest-relative-root>/
  jobs/         private job state and artifacts
  tmp/          private upload staging
```

The four JSON files are mandatory. `context.json` uses
`specstyle.production.context.v1`; the other files use the
`specstyle.production.models.v1`,
`specstyle.production.weight_manifests.v1`, and
`specstyle.production.license_approvals.v1` schemas. Every evidence digest in
`context.json` must resolve to a regular content-addressed file at the path
shown above. Every model descriptor, manifest, approval, local file hash, role,
revision, and license identifier must close over the same local supply.

These files are not generic examples: model manifests depend on the exact local
weight bytes, license approval is a human decision, and calibrated threshold
evidence must come from the approved annotation workflow. The repository does
not currently contain a fresh-runtime supply/calibration package generator.
Copy an independently approved package into these paths before launching; the
bootstrap intentionally fails closed when any file, approval, hash, permission,
or evidence object is absent. Do not reuse the engineering smoke threshold
markers as calibration evidence.

The `config/` and `evidence/` directories, the four JSON files, and every
evidence object must be owned by the service process's effective user. The
runtime root and `models/` may be owned by root or that service user. None of
these paths may be group- or world-writable. Mutable job directories are
created with mode `0700`; the four config files and each evidence object must
use mode `0600`.

## Launch the UI

Local-only mode does not require application credentials:

```bash
cd /path/to/lookfile
PYTHONPATH="$PWD/src" /opt/venv/bin/python \
  deployment/amd/scripts/launch_production_ui.py \
  --runtime-root /workspace/persistence/lookfile-runtime \
  --host 127.0.0.1 \
  --port 7860
```

Binding to `0.0.0.0` requires both credentials. Use it only behind the AMD
platform's HTTPS reverse proxy:

```bash
export SPECSTYLE_UI_USERNAME='<username>'
export SPECSTYLE_UI_PASSWORD='<at-least-16-characters>'
PYTHONPATH="$PWD/src" /opt/venv/bin/python \
  deployment/amd/scripts/launch_production_ui.py \
  --runtime-root /workspace/persistence/lookfile-runtime \
  --host 0.0.0.0 \
  --port 7860
```

The UI returns only published bundle paths. Gradio allowlists the export root;
config, evidence, weights, job state, materialized artifacts, style assets, and
upload staging are explicitly blocked.

## Validation

With the pinned dependencies available:

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q
ruff check .
ruff format --check .
```

CPU unit tests use deterministic fakes. Production tests validate supply,
filesystem, workflow, routing, cleanup, and export contracts without downloading
weights. Radeon smoke evidence must be collected separately on the target AMD
runtime.

## Security and licensing

- Never commit model weights, credentials, private images, or unlicensed media.
- A code license does not grant a model-weight license. Every Production model
  requires an independently recorded revision, content hash, SPDX identifier,
  and approval evidence before it can load.
- Failed required gates route to `rejected/`; manual decisions remain separate;
  only the automatic whitelist state can enter `approved/`.
- ControlNet preserves a spatial condition. It does not establish identity.
