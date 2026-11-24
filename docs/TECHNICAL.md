# WeightCreator v1: implementation

## Inputs and boundaries

Policy inputs are RGB, metric depth, camera intrinsics/extrinsics, end-effector pose, gripper proprioception, and instruction text. Simulator task IDs, initial-state arrays and native success are evaluator-only data. No simulator object pose, segmentation identity, hidden articulation state or goal predicate is supplied to the policy. Known robot hand geometry is used for visible-point exclusion and collision screening.

## Semantic grounding

`perception.py` estimates the dominant support plane, discretizes calibrated XYZ into 4 mm voxels and uses 26-connected components. Candidate bounds use the 2nd/98th percentiles. Layered proposals handle relational scenes. Candidate IDs identify observed regions, not simulator instances.

`semantic_grounding.py` asks whether each crop depicts the requested noun phrase. The score is logit(Yes) minus logit(No), obtained from one frozen forward pass with thinking disabled. `semantic_joint.py` associates primary and wrist views and assigns distinct regions to multiple source objects. `relation_v7` uses the maximum available view margin and combines semantic evidence with distance, between-segment deviation, height and shape penalties. Robot-overlapping regions are suppressed using observed robot pose.

Qwen hidden states are not fed into a learned action decoder. Parsing and routing are explicit Python grammar rules. Qwen provides visual semantic evidence, rather than a generated trajectory or general-purpose plan.

## Recurrent motor expert

`WeightActionExpert` registers all checkpoint tensors as non-trainable parameters. Its condition is c=[source_xyz,target_xyz,1], after geometric corrections. The phase is an eight-dimensional one-hot state. Target weights have shape [8,3,7]; the active affine map produces a Cartesian target.

The phases are approach, descend, close, lift, transfer, lower, release, retreat. Position and axis-angle rotation errors form a six-dimensional vector. A diagonal servo matrix diag(12,12,12,4,4,4), phase-specific clipping and a gripper scalar produce the native seven-dimensional action. Native robot control executes this command. These command values are not literal position increments in meters.

Phase transitions require positional and, where configured, rotational proximity for a specified dwell duration. The dwell counter resets when proximity is lost. Closing requires 20 qualifying steps and release 10. Native success does not control transitions. Panda end-effector observations use the body frame, with Y as the jaw-closing axis.

`ProgramActionExpert` adds first-object / second-object / done states and supports one or two pick/place primitives. It advances after retreat completion. Shared containers receive separate size-scaled placement slots. `SequenceActionExpert` connects twisting and pick/place skills from motor completion signals.

## Current geometry

* Bowls: eight authored rim-grasp candidates, visible hand/finger collision features and view preference.
* Tall non-bowl objects: grasp approximately 30% below the observed top; descend/close tolerances 12/18 mm.
* Support surfaces: local visible depth estimates carrying/release height rather than using an enclosing bounding-box top.
* Inclined surfaces: tall objects can be laid along the observed uphill direction with corresponding target correction.
* Payload correction: phase-boundary RGB-D estimates the carried body's center and compensates an off-center grip.
* Relative placement: three yaw candidates and 0/30/60 mm additional outward clearances are scored against visible obstacles.
* v52 bowl reachability: lift and transfer height are capped by the actually attained approach height. This addition lacks a completed broad validation at release time and may affect obstacle clearance.

These are finite geometric candidate checks, not a globally collision-free trajectory planner. Occlusion, clipped container bounds, narrow cavities, unsupported compositions and recovery after a failed grasp remain limitations.

## Checkpoint bank

| Checkpoint | Scalars | Function |
|---|---:|---|
| expert_v52.pt | 499 | Pick/place and two-object program |
| relation_v7.pt | 76 | Semantic/geometric relations |
| fixture_v1.pt | 363 | Drawer opening |
| twist_v1.pt | 296 | Stove handle rotation |
| push_v2.pt | 359 | Planar pushing |
| sequence_v0.pt | 10 | Skill sequencing |
| Total | 1603 | Excludes frozen VLM and software infrastructure |

Independent insertion and drawer-composition research branches are not integrated into this release's unified routing and are not advertised as supported full tasks. Tensor counts do not capture the complexity of Python perception, geometry, grammar or the native controller.
