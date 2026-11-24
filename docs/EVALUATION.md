# Evaluation and evidence

The full v1 protocol covers LIBERO-Spatial, Object, Goal and Long (`libero_10`), ten tasks each and all 50 supplied initial states per task. Native action limits are respectively 220, 280, 300 and 520 steps. Initial capture uses both existing cameras at 1024 pixels; rollout follows the released implementation. Every episode obtains fresh observations and semantic grounding.

Primary metric: final native success within the suite's original step limit. Report release, success over the last ten steps and complete motor-program termination separately. Unsupported instructions and normal native failures remain in the denominator. Infrastructure failures are separately recorded and cannot be silently converted into completed business results or omitted to improve the score.

The full evaluation includes previously used development states: it is not an unseen-state claim. Worker metadata maps the fixture CLI's existing `frozen-heldout` argument to the public `frozen-full-evaluation` cohort; that legacy string does not imply these states were held out.

Workers use atomic mkdir claims, preserve status/result/trace files, validate policy hashes and do not repeat completed normal failures. Run one aggregation writer. Platform success additionally requires successful outer/worker exits and terminal platform confirmation, beyond business metrics.

Historical v36 audited results, using 80 episodes at states 33/34:

| Suite | Native success | Released success | Complete-program success |
|---|---:|---:|---:|
| Spatial | 17/20 | 17 | 13 |
| Object | 16/20 | 16 | 16 |
| Goal | 15/20 | 14 | 12 |
| Long | 9/20 | 9 | 9 |
| Total | 57/80 | 56 | 50 |

All 57 native successes held over the last ten steps; infrastructure errors were zero. These numbers belong to v36 and must not be attributed to v1/v52. The v1 goal is at least 85% overall with a per-suite breakdown; no final v1 result is asserted here.
