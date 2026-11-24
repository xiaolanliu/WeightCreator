# Release provenance

Public name: **WeightCreator**. Public version: **v1 / 1.0.0**. Internal policy revision: **expert_v52**.

Core policy source and six checkpoint files originate from the frozen full-evaluation snapshot. The public packaging replaces a machine-specific inference import with `backbone.py`; model class, local-only loading, BF16, SDPA and CUDA device placement retain the original behavior. No model weights or action-control algorithm were changed for publication.

`release_manifest.json` binds distributed files to SHA-256 digests and records original core source hashes. The historical direct-weight development process reconstructed and checked 83 checkpoint revisions locally; this release distributes the six active checkpoints, not all 83 development branches. This package does not claim to reproduce the full historical revision ledger.

The original SCO evaluation is independent of this portable export and continues with its frozen snapshot. Public environment instructions are a migration guide: the portable package has static/import-path verification, while end-to-end native execution was verified in the original SCO environments. Dependency installation on a new host requires its own validation.

Qwen and LIBERO are external projects with their own licenses. This release does not redistribute their weights or assets. No new open-source license grant is inferred for the author's code; the owner may select a license separately.
