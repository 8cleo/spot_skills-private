"""Collect validated, object-centric state transitions from Spot skill executions.

This module deliberately contains no ROS imports.  The physical-state observation boundary is the
``capture_state`` callback supplied by the Spot integration layer.  Keeping the orchestration
independent of ROS makes the safety, ordering, and persistence behavior testable without a robot.

A physical reset starts a new trace.  Within a trace, each committed transition contains a fresh
pre-state, the fully resolved skill invocation, the robot and operator outcomes, and a fresh
post-state.  A directory is published only after all three YAML files validate successfully.
"""

from __future__ import annotations

import shutil
import time
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence
from uuid import uuid4

from robotics_utils.io.pydantic_schemata import ImageObservationSchema, Pose3DDictSchema
from robotics_utils.io.yaml_utils import export_yaml_data
from robotics_utils.skills import (
    Outcome,
    SkillInstance,
    SkillParamKey,
    SkillsInventory,
    SkillsProtocol,
    find_default_param_values,
)
from robotics_utils.spatial import Pose3D
from spot_skills_py.spot_collection_schemas import (
    BooleanArgumentSchema,
    ContinuousArgumentSchema,
    ExecutedSkillSchema,
    ExecutionOutcomeSchema,
    IntegerArgumentSchema,
    ObjectArgumentSchema,
    PathArgumentSchema,
    PoseArgumentSchema,
    SkillArgumentSchema,
    StateSnapshotSchema,
    StringArgumentSchema,
    TransitionRecordSchema,
    WaypointArgumentSchema,
    load_schema_yaml,
    save_schema_yaml,
)


ArgumentKind = Literal[
    "object",
    "waypoint",
    "pose",
    "continuous",
    "integer",
    "boolean",
    "path",
    "string",
]
SnapshotPhase = Literal["before", "after"]
ExecutionStatus = Literal["completed", "exception", "interrupted"]
OperatorVerdict = Literal["success", "failure", "uncertain", "not_reviewed"]


# SkillsInventory includes every @skill_method, including private implementation helpers.  This
# allowlist intentionally contains only the macro actions that are plausible operator-learning
# examples for the current Spot experiment.  Skills with implicit symbolic targets (for example,
# OpenDoor without a door argument) remain excluded until their signatures identify those targets.
# Callers can provide a narrower or explicitly reviewed alternative set.
DEFAULT_COLLECTABLE_SPOT_SKILLS = frozenset(
    {
        "NavigateToWaypoint",
        "OpenDrawer",
        "Pick",
        "PickFromDrawer",
        "PickFromFilingCabinet",
        "Place",
    },
)


# Python annotates object names, waypoints, and arbitrary text as ``str``.  These explicit tags
# preserve the distinction needed by the abstraction-learning pipeline.  Numeric, Boolean, Path,
# and Pose3D arguments are inferred from their runtime types.
DEFAULT_SPOT_ARGUMENT_KINDS: dict[SkillParamKey, ArgumentKind] = {
    ("NavigateToWaypoint", "waypoint"): "waypoint",
    ("OpenDrawer", "container_name"): "object",
    ("Pick", "object_name"): "object",
    ("Pick", "ignored_objects"): "string",
    ("PickFromDrawer", "object_name"): "object",
    ("PickFromDrawer", "drawer_name"): "object",
    ("PickFromFilingCabinet", "object_name"): "object",
    ("PickFromFilingCabinet", "cabinet_name"): "object",
    ("PickFromFilingCabinet", "ignored_objects"): "string",
    ("Place", "object_name"): "object",
    ("Place", "surface_name"): "object",
    ("PlaceOnCabinet", "object_name"): "object",
}


@dataclass(frozen=True)
class SkillExecutionSpec:
    """A requested skill and any argument values that override its Python defaults."""

    skill_name: str
    bindings: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedSkill:
    """A fully validated runtime skill instance and its durable representation."""

    instance: SkillInstance
    action: ExecutedSkillSchema


@dataclass(frozen=True)
class SnapshotCaptureRequest:
    """Paths and context supplied to the live Spot state-observation adapter.

    Sensor artifacts must be written beneath ``asset_output_directory``.  Their corresponding
    schema paths must begin with ``asset_relative_directory`` because the staging transition is
    later renamed into its final location under ``dataset_root``.
    """

    snapshot_id: str
    phase: SnapshotPhase
    dataset_root: Path
    snapshot_relative_path: Path
    transition_output_directory: Path
    asset_output_directory: Path
    asset_relative_directory: Path
    executed_action: ExecutedSkillSchema | None


@dataclass(frozen=True)
class OperatorReview:
    """Human assessment and curation decision for one physical action attempt."""

    verdict: OperatorVerdict
    notes: str | None = None
    accepted_for_learning: bool = True


@dataclass(frozen=True)
class ReviewRequest:
    """Information shown to, or otherwise consumed by, an operator reviewer."""

    action: ExecutedSkillSchema
    status: ExecutionStatus
    robot_success: bool | None
    robot_message: str
    before: StateSnapshotSchema
    after: StateSnapshotSchema


StateCapture = Callable[[SnapshotCaptureRequest], StateSnapshotSchema]
OperatorReviewer = Callable[[ReviewRequest], OperatorReview]
ExecutionApprover = Callable[[ExecutedSkillSchema], bool]
Clock = Callable[[], float]
IdFactory = Callable[[], str]
SchemaWriter = Callable[[object, Path], None]


class TraceNotUsableError(RuntimeError):
    """Raised when collection cannot safely continue in the current physical trace."""


class CollectionAttemptError(RuntimeError):
    """Raised when an attempted transition cannot be committed as trustworthy data."""

    def __init__(
        self,
        message: str,
        *,
        action_started: bool,
        attempt_directory: Path | None,
    ) -> None:
        """Initialize an error with physical-execution and recovery context."""
        super().__init__(message)
        self.action_started = action_started
        self.attempt_directory = attempt_directory


def _new_identifier() -> str:
    """Return a collision-resistant identifier suitable for a dataset filename."""
    return uuid4().hex


def _validate_path_component(value: str, label: str) -> str:
    """Reject empty identifiers and identifiers that could change directory structure."""
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be a non-empty filename component, got '{value}'.")
    return value


def _require_relative_path(path: Path, label: str) -> Path:
    """Require a portable dataset-relative artifact path."""
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be dataset-relative, got '{path}'.")
    return path


def _relative_suffix(path: Path, parent: Path, label: str) -> Path:
    """Return a path's suffix beneath a required parent directory."""
    try:
        return path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{label} '{path}' must be beneath '{parent}'.") from exc


def resolve_skill_instance(
    inventory: SkillsInventory,
    default_values: Mapping[SkillParamKey, object],
    allowed_skills: frozenset[str],
    spec: SkillExecutionSpec,
) -> SkillInstance:
    """Resolve defaults and validate every binding before any physical execution.

    Direct construction of ``SkillInstance`` performs no validation.  This function guarantees
    that the saved action and the executed action contain the same complete parameter mapping.
    """
    if spec.skill_name not in allowed_skills:
        raise ValueError(
            f"Skill '{spec.skill_name}' is not enabled for collection. Allowed skills: "
            f"{sorted(allowed_skills)}.",
        )

    skill = inventory.get_skill(spec.skill_name)
    parameter_names = {parameter.name for parameter in skill.parameters}
    extra_names = set(spec.bindings).difference(parameter_names)
    if extra_names:
        raise ValueError(
            f"Skill '{skill.name}' received unknown bindings: {sorted(extra_names)}.",
        )

    resolved_bindings: dict[str, object] = {}
    for parameter in skill.parameters:
        key = (skill.name, parameter.name)
        if parameter.name in spec.bindings:
            value = spec.bindings[parameter.name]
        elif key in default_values:
            value = deepcopy(default_values[key])
        else:
            raise ValueError(
                f"Skill '{skill.name}' requires a binding for '{parameter.name}'.",
            )

        if not isinstance(value, parameter.type_):
            raise TypeError(
                f"Skill '{skill.name}' parameter '{parameter.name}' expects "
                f"{parameter.type_name}, received {type(value).__name__}: {value!r}.",
            )
        if parameter.type_ is int and isinstance(value, bool):
            raise TypeError(
                f"Skill '{skill.name}' integer parameter '{parameter.name}' cannot be Boolean.",
            )

        resolved_bindings[parameter.name] = value

    return SkillInstance(skill=skill, bindings=resolved_bindings)


def _infer_argument_kind(value: object) -> ArgumentKind:
    """Infer an unambiguous serialized argument kind from a runtime value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Integral):
        return "integer"
    if isinstance(value, Real):
        return "continuous"
    if isinstance(value, Path):
        return "path"
    if isinstance(value, (Pose3D, Pose3DDictSchema)):
        return "pose"
    if isinstance(value, str):
        raise ValueError(
            "String skill arguments are semantically ambiguous. Configure the argument as "
            "'object', 'waypoint', or 'string'.",
        )
    raise TypeError(f"Unsupported skill argument type: {type(value).__name__}.")


def _serialize_argument(name: str, kind: ArgumentKind, value: object) -> SkillArgumentSchema:
    """Convert one resolved runtime binding into its tagged persistence schema."""
    if kind in {"object", "waypoint", "string"}:
        if not isinstance(value, str):
            raise TypeError(f"Argument '{name}' tagged '{kind}' must be a string.")
        if kind == "object":
            return ObjectArgumentSchema(name=name, kind=kind, value=value)
        if kind == "waypoint":
            return WaypointArgumentSchema(name=name, kind=kind, value=value)
        return StringArgumentSchema(name=name, kind=kind, value=value)

    if kind == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"Argument '{name}' tagged 'boolean' must be Boolean.")
        return BooleanArgumentSchema(name=name, kind=kind, value=value)

    if kind == "integer":
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"Argument '{name}' tagged 'integer' must be an integer.")
        return IntegerArgumentSchema(name=name, kind=kind, value=int(value))

    if kind == "continuous":
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"Argument '{name}' tagged 'continuous' must be a real number.")
        return ContinuousArgumentSchema(name=name, kind=kind, value=float(value))

    if kind == "path":
        if not isinstance(value, Path):
            raise TypeError(f"Argument '{name}' tagged 'path' must be a pathlib.Path.")
        return PathArgumentSchema(name=name, kind=kind, value=value)

    if kind == "pose":
        if isinstance(value, Pose3D):
            pose_schema = value.to_schema()
        elif isinstance(value, Pose3DDictSchema):
            pose_schema = value
        else:
            raise TypeError(f"Argument '{name}' tagged 'pose' must be Pose3D.")

        if not isinstance(pose_schema, Pose3DDictSchema):
            raise TypeError(f"Argument '{name}' pose must preserve an explicit reference frame.")
        return PoseArgumentSchema(name=name, kind=kind, value=pose_schema)

    raise ValueError(f"Unsupported argument kind '{kind}' for '{name}'.")


def serialize_skill_instance(
    instance: SkillInstance,
    argument_kinds: Mapping[SkillParamKey, ArgumentKind],
) -> ExecutedSkillSchema:
    """Serialize every resolved binding in the skill's formal parameter order."""
    serialized_arguments: list[SkillArgumentSchema] = []
    for parameter in instance.skill.parameters:
        if parameter.name not in instance.bindings:
            raise ValueError(
                f"Cannot serialize unresolved parameter '{instance.skill.name}.{parameter.name}'.",
            )

        value = instance.bindings[parameter.name]
        kind = argument_kinds.get((instance.skill.name, parameter.name))
        if kind is None:
            try:
                kind = _infer_argument_kind(value)
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Cannot serialize '{instance.skill.name}.{parameter.name}': {exc}",
                ) from exc

        serialized_arguments.append(_serialize_argument(parameter.name, kind, value))

    return ExecutedSkillSchema(name=instance.skill.name, arguments=serialized_arguments)


def prompt_to_execute(action: ExecutedSkillSchema) -> bool:
    """Ask the operator whether a prepared physical action should be executed."""
    from rich.prompt import Confirm

    arguments = ", ".join(f"{argument.name}={argument.value}" for argument in action.arguments)
    signature = f"{action.name}({arguments})"
    return Confirm.ask(f"Execute [bold]{signature}[/]?", default=True)


def prompt_for_operator_review(request: ReviewRequest) -> OperatorReview:
    """Collect a human success label after inspecting the physical post-state."""
    from rich.prompt import Confirm, Prompt

    if request.robot_success is True:
        default_verdict = "success"
    elif request.robot_success is False:
        default_verdict = "failure"
    else:
        default_verdict = "uncertain"

    verdict = Prompt.ask(
        "Did the intended physical skill effect occur?",
        choices=["success", "failure", "uncertain", "not_reviewed"],
        default=default_verdict,
    )
    notes_text = Prompt.ask("Operator notes (optional)", default="").strip()
    accepted = Confirm.ask(
        "Keep this transition for learning?",
        default=verdict != "uncertain",
    )
    return OperatorReview(
        verdict=verdict,
        notes=notes_text or None,
        accepted_for_learning=accepted,
    )


class SpotTransitionCollector:
    """Execute prepared Spot skills and atomically persist their state transitions.

    Use ``new_trace`` after physically resetting the environment.  Use ``resume_trace`` only when
    the environment has not been reset or changed since the last committed transition.
    """

    _TAINT_FILENAME = "TRACE_TAINTED.yaml"

    def __init__(
        self,
        *,
        executor: SkillsProtocol,
        capture_state: StateCapture,
        dataset_root: Path,
        trace_id: str,
        trace_directory: Path,
        next_transition_index: int,
        allowed_skills: frozenset[str],
        argument_kinds: Mapping[SkillParamKey, ArgumentKind],
        reviewer: OperatorReviewer,
        clock: Clock,
        id_factory: IdFactory,
        schema_writer: SchemaWriter,
    ) -> None:
        """Initialize a collector after its trace directory has been checked or created."""
        self._executor = executor
        self._capture_state = capture_state
        self.dataset_root = dataset_root.resolve()
        self.trace_id = trace_id
        self._trace_directory = trace_directory
        self._transitions_directory = trace_directory / "transitions"
        self._next_transition_index = next_transition_index
        self._allowed_skills = allowed_skills
        self._argument_kinds = dict(argument_kinds)
        self._reviewer = reviewer
        self._clock = clock
        self._id_factory = id_factory
        self._schema_writer = schema_writer
        self._inventory = SkillsInventory.from_protocol(type(executor))
        self._default_values = find_default_param_values(type(executor))
        self._tainted = False

    @classmethod
    def new_trace(
        cls,
        *,
        executor: SkillsProtocol,
        capture_state: StateCapture,
        dataset_root: Path,
        trace_id: str,
        allowed_skills: frozenset[str] = DEFAULT_COLLECTABLE_SPOT_SKILLS,
        argument_kinds: Mapping[SkillParamKey, ArgumentKind] | None = None,
        reviewer: OperatorReviewer = prompt_for_operator_review,
        clock: Clock = time.time,
        id_factory: IdFactory = _new_identifier,
        schema_writer: SchemaWriter = save_schema_yaml,
    ) -> SpotTransitionCollector:
        """Create an empty trace after the operator has physically reset the world."""
        trace_id = _validate_path_component(trace_id, "trace_id")
        dataset_root = dataset_root.resolve()
        trace_directory = dataset_root / "traces" / trace_id
        trace_directory.parent.mkdir(parents=True, exist_ok=True)
        trace_directory.mkdir(exist_ok=False)
        (trace_directory / "transitions").mkdir()

        return cls(
            executor=executor,
            capture_state=capture_state,
            dataset_root=dataset_root,
            trace_id=trace_id,
            trace_directory=trace_directory,
            next_transition_index=0,
            allowed_skills=allowed_skills,
            argument_kinds=(
                DEFAULT_SPOT_ARGUMENT_KINDS if argument_kinds is None else argument_kinds
            ),
            reviewer=reviewer,
            clock=clock,
            id_factory=id_factory,
            schema_writer=schema_writer,
        )

    @classmethod
    def resume_trace(
        cls,
        *,
        executor: SkillsProtocol,
        capture_state: StateCapture,
        dataset_root: Path,
        trace_id: str,
        allowed_skills: frozenset[str] = DEFAULT_COLLECTABLE_SPOT_SKILLS,
        argument_kinds: Mapping[SkillParamKey, ArgumentKind] | None = None,
        reviewer: OperatorReviewer = prompt_for_operator_review,
        clock: Clock = time.time,
        id_factory: IdFactory = _new_identifier,
        schema_writer: SchemaWriter = save_schema_yaml,
    ) -> SpotTransitionCollector:
        """Resume a complete, contiguous, untainted trace without resetting the world."""
        trace_id = _validate_path_component(trace_id, "trace_id")
        dataset_root = dataset_root.resolve()
        trace_directory = dataset_root / "traces" / trace_id
        transitions_directory = trace_directory / "transitions"

        if not transitions_directory.is_dir():
            raise FileNotFoundError(f"Trace '{trace_id}' does not exist in {dataset_root}.")
        if (trace_directory / cls._TAINT_FILENAME).exists():
            raise TraceNotUsableError(f"Trace '{trace_id}' is marked as tainted.")

        incomplete = [path for path in transitions_directory.iterdir() if path.name.startswith(".")]
        if incomplete:
            raise TraceNotUsableError(
                f"Trace '{trace_id}' contains incomplete attempts: "
                f"{[path.name for path in incomplete]}.",
            )

        records: list[TransitionRecordSchema] = []
        for transition_directory in transitions_directory.iterdir():
            if not transition_directory.is_dir():
                raise TraceNotUsableError(
                    f"Unexpected file in trace transitions: {transition_directory}.",
                )

            record_path = transition_directory / "transition.yaml"
            if not record_path.is_file():
                raise TraceNotUsableError(
                    f"Committed transition is missing its record: {transition_directory}.",
                )

            record = load_schema_yaml(TransitionRecordSchema, record_path)
            if record.trace_id != trace_id:
                raise TraceNotUsableError(
                    f"Transition '{record.transition_id}' belongs to trace '{record.trace_id}', "
                    f"not '{trace_id}'.",
                )

            _validate_path_component(record.transition_id, "transition_id")
            expected_directory_name = f"{record.transition_index:06d}-{record.transition_id}"
            if transition_directory.name != expected_directory_name:
                raise TraceNotUsableError(
                    f"Transition directory '{transition_directory.name}' should be "
                    f"'{expected_directory_name}'.",
                )

            expected_relative_directory = (
                Path("traces") / trace_id / "transitions" / expected_directory_name
            )
            expected_before_path = expected_relative_directory / "before.yaml"
            expected_after_path = expected_relative_directory / "after.yaml"
            if record.before_state_path != expected_before_path:
                raise TraceNotUsableError(
                    f"Transition '{record.transition_id}' has unexpected pre-state path "
                    f"'{record.before_state_path}'.",
                )
            if record.after_state_path != expected_after_path:
                raise TraceNotUsableError(
                    f"Transition '{record.transition_id}' has unexpected post-state path "
                    f"'{record.after_state_path}'.",
                )

            before = load_schema_yaml(StateSnapshotSchema, dataset_root / expected_before_path)
            after = load_schema_yaml(StateSnapshotSchema, dataset_root / expected_after_path)
            if before.snapshot_id != f"{record.transition_id}-before":
                raise TraceNotUsableError(
                    f"Transition '{record.transition_id}' has mismatched pre-state ID "
                    f"'{before.snapshot_id}'.",
                )
            if after.snapshot_id != f"{record.transition_id}-after":
                raise TraceNotUsableError(
                    f"Transition '{record.transition_id}' has mismatched post-state ID "
                    f"'{after.snapshot_id}'.",
                )

            cls._validate_committed_snapshot_artifacts(
                before,
                dataset_root=dataset_root,
                required_image_directory=(
                    expected_relative_directory / "observations" / "before"
                ),
            )
            cls._validate_committed_snapshot_artifacts(
                after,
                dataset_root=dataset_root,
                required_image_directory=(
                    expected_relative_directory / "observations" / "after"
                ),
            )
            records.append(record)

        indices = sorted(record.transition_index for record in records)
        if indices != list(range(len(records))):
            raise TraceNotUsableError(
                f"Trace '{trace_id}' indices must be contiguous from zero, got {indices}.",
            )

        return cls(
            executor=executor,
            capture_state=capture_state,
            dataset_root=dataset_root,
            trace_id=trace_id,
            trace_directory=trace_directory,
            next_transition_index=len(records),
            allowed_skills=allowed_skills,
            argument_kinds=(
                DEFAULT_SPOT_ARGUMENT_KINDS if argument_kinds is None else argument_kinds
            ),
            reviewer=reviewer,
            clock=clock,
            id_factory=id_factory,
            schema_writer=schema_writer,
        )

    @property
    def next_transition_index(self) -> int:
        """Return the index that the next successfully committed transition will receive."""
        return self._next_transition_index

    def prepare(self, spec: SkillExecutionSpec) -> PreparedSkill:
        """Validate, resolve, and serialize a skill without executing it."""
        instance = resolve_skill_instance(
            self._inventory,
            self._default_values,
            self._allowed_skills,
            spec,
        )
        action = serialize_skill_instance(instance, self._argument_kinds)
        return PreparedSkill(instance=instance, action=action)

    def collect_transition(self, spec: SkillExecutionSpec) -> TransitionRecordSchema:
        """Collect one explicitly selected transition without an execution prompt."""
        return self._collect_prepared(self.prepare(spec))

    def collect_trace(
        self,
        steps: Sequence[SkillExecutionSpec],
        *,
        approve_execution: ExecutionApprover = prompt_to_execute,
    ) -> list[TransitionRecordSchema]:
        """Collect one explicit, ordered sequence of skills in the current physical trace.

        A physical reset requires a new ``SpotTransitionCollector.new_trace`` call.  Randomization
        and repetition belong in a higher-level trial coordinator that performs that reset between
        traces; doing either inside this method could violate action preconditions.  Declining a
        step records no transition because no physical attempt occurred.
        """
        prepared_steps = [self.prepare(step) for step in steps]
        records: list[TransitionRecordSchema] = []
        for prepared in prepared_steps:
            if approve_execution(prepared.action):
                records.append(self._collect_prepared(prepared))
        return records

    def _assert_trace_usable(self) -> None:
        """Refuse to hide an unrecorded physical state change with another transition."""
        taint_path = self._trace_directory / self._TAINT_FILENAME
        if self._tainted or taint_path.exists():
            raise TraceNotUsableError(
                f"Trace '{self.trace_id}' is tainted. Physically reset and start a new trace.",
            )

        incomplete = [
            path for path in self._transitions_directory.iterdir() if path.name.startswith(".")
        ]
        if incomplete:
            raise TraceNotUsableError(
                f"Trace '{self.trace_id}' contains incomplete attempts: "
                f"{[path.name for path in incomplete]}.",
            )

    @staticmethod
    def _validate_snapshot_artifacts(
        snapshot: StateSnapshotSchema,
        *,
        artifact_path: Callable[[Path], Path],
        required_image_directory: Path,
    ) -> None:
        """Require every file referenced by a snapshot to exist and remain portable."""
        environment_config = _require_relative_path(
            snapshot.environment_config,
            "environment_config",
        )
        if not artifact_path(environment_config).is_file():
            raise FileNotFoundError(
                f"Environment configuration does not exist: '{environment_config}'.",
            )

        if snapshot.occupancy_grid_path is not None:
            occupancy_path = _require_relative_path(
                snapshot.occupancy_grid_path,
                "occupancy_grid_path",
            )
            if not artifact_path(occupancy_path).is_file():
                raise FileNotFoundError(f"Occupancy grid does not exist: '{occupancy_path}'.")

        for obj in snapshot.objects:
            for observation_path in obj.image_observations:
                observation_path = _require_relative_path(
                    observation_path,
                    f"Image observation for '{obj.name}'",
                )
                _relative_suffix(
                    observation_path,
                    required_image_directory,
                    f"Image observation for '{obj.name}'",
                )
                physical_observation_path = artifact_path(observation_path)
                if not physical_observation_path.is_file():
                    raise FileNotFoundError(
                        f"Image observation does not exist: '{observation_path}'.",
                    )

                observation = load_schema_yaml(
                    ImageObservationSchema,
                    physical_observation_path,
                )
                image_path = _require_relative_path(
                    observation.image_path,
                    f"Image referenced by '{observation_path}'",
                )
                _relative_suffix(
                    image_path,
                    required_image_directory,
                    f"Image referenced by '{observation_path}'",
                )
                if not artifact_path(image_path).is_file():
                    raise FileNotFoundError(f"Observed image does not exist: '{image_path}'.")

    @classmethod
    def _validate_committed_snapshot_artifacts(
        cls,
        snapshot: StateSnapshotSchema,
        *,
        dataset_root: Path,
        required_image_directory: Path,
    ) -> None:
        """Validate artifact references in a transition that is already committed."""

        def committed_path(relative_path: Path) -> Path:
            return dataset_root / relative_path

        cls._validate_snapshot_artifacts(
            snapshot,
            artifact_path=committed_path,
            required_image_directory=required_image_directory,
        )

    def _validate_staged_snapshot_artifacts(
        self,
        snapshot: StateSnapshotSchema,
        request: SnapshotCaptureRequest,
    ) -> None:
        """Validate artifact references while the current transition is still staged."""
        final_transition_directory = request.snapshot_relative_path.parent

        def staged_path(relative_path: Path) -> Path:
            try:
                transition_suffix = relative_path.relative_to(final_transition_directory)
            except ValueError:
                return request.dataset_root / relative_path
            return request.transition_output_directory / transition_suffix

        self._validate_snapshot_artifacts(
            snapshot,
            artifact_path=staged_path,
            required_image_directory=request.asset_relative_directory,
        )

    def _capture_snapshot(
        self,
        *,
        phase: SnapshotPhase,
        snapshot_id: str,
        staging_directory: Path,
        final_relative_directory: Path,
        executed_action: ExecutedSkillSchema | None,
    ) -> tuple[StateSnapshotSchema, Path]:
        """Capture and stage one state snapshot using the injected live-state adapter."""
        snapshot_relative_path = final_relative_directory / f"{phase}.yaml"
        asset_relative_directory = final_relative_directory / "observations" / phase
        asset_output_directory = staging_directory / "observations" / phase
        asset_output_directory.mkdir(parents=True)

        request = SnapshotCaptureRequest(
            snapshot_id=snapshot_id,
            phase=phase,
            dataset_root=self.dataset_root,
            snapshot_relative_path=snapshot_relative_path,
            transition_output_directory=staging_directory,
            asset_output_directory=asset_output_directory,
            asset_relative_directory=asset_relative_directory,
            executed_action=executed_action,
        )
        snapshot = self._capture_state(request)
        if not isinstance(snapshot, StateSnapshotSchema):
            raise TypeError(
                "capture_state must return StateSnapshotSchema, received "
                f"{type(snapshot).__name__}.",
            )
        if snapshot.snapshot_id != snapshot_id:
            raise ValueError(
                f"capture_state returned snapshot_id '{snapshot.snapshot_id}', expected "
                f"'{snapshot_id}'.",
            )

        self._validate_staged_snapshot_artifacts(snapshot, request)
        staging_path = staging_directory / f"{phase}.yaml"
        self._schema_writer(snapshot, staging_path)
        return snapshot, staging_path

    def _collect_prepared(self, prepared: PreparedSkill) -> TransitionRecordSchema:
        """Execute and commit a skill that has already passed all preflight checks."""
        self._assert_trace_usable()

        transition_id = _validate_path_component(self._id_factory(), "transition_id")
        transition_index = self._next_transition_index
        directory_name = f"{transition_index:06d}-{transition_id}"
        final_relative_directory = Path("traces") / self.trace_id / "transitions" / directory_name
        final_directory = self.dataset_root / final_relative_directory
        staging_directory = self._transitions_directory / f".{directory_name}.incomplete"

        if final_directory.exists() or staging_directory.exists():
            raise FileExistsError(f"Transition destination already exists: {directory_name}.")
        staging_directory.mkdir()

        stage = "capturing pre-state"
        try:
            before, before_staging_path = self._capture_snapshot(
                phase="before",
                snapshot_id=f"{transition_id}-before",
                staging_directory=staging_directory,
                final_relative_directory=final_relative_directory,
                executed_action=None,
            )
        except (Exception, KeyboardInterrupt) as exc:
            shutil.rmtree(staging_directory, ignore_errors=True)
            raise CollectionAttemptError(
                f"Collection failed before physical execution while {stage}: {exc}",
                action_started=False,
                attempt_directory=None,
            ) from exc

        action_started = False
        execution_interrupt: KeyboardInterrupt | None = None
        review_interrupt: KeyboardInterrupt | None = None

        try:
            stage = "executing skill"
            started_at_s = self._clock()
            action_started = True
            try:
                result = prepared.instance.execute(self._executor)
                if not isinstance(result, Outcome):
                    raise TypeError(
                        "Collected skills must return robotics_utils.skills.Outcome, received "
                        f"{type(result).__name__}.",
                    )
                status: ExecutionStatus = "completed"
                robot_success: bool | None = result.success
                robot_message = result.message
            except KeyboardInterrupt as exc:
                execution_interrupt = exc
                status = "interrupted"
                robot_success = None
                robot_message = "Skill execution interrupted by the operator."
            except Exception as exc:  # A physical attempt with an exception remains useful data.
                status = "exception"
                robot_success = None
                robot_message = f"{type(exc).__name__}: {exc}"
            finished_at_s = self._clock()

            stage = "capturing post-state"
            after, after_staging_path = self._capture_snapshot(
                phase="after",
                snapshot_id=f"{transition_id}-after",
                staging_directory=staging_directory,
                final_relative_directory=final_relative_directory,
                executed_action=prepared.action,
            )

            stage = "reviewing physical outcome"
            review_request = ReviewRequest(
                action=prepared.action,
                status=status,
                robot_success=robot_success,
                robot_message=robot_message,
                before=before,
                after=after,
            )
            try:
                review = self._reviewer(review_request)
            except KeyboardInterrupt as exc:
                review_interrupt = exc
                review = OperatorReview(
                    verdict="not_reviewed",
                    notes="Operator review was interrupted.",
                    accepted_for_learning=False,
                )
            except Exception as exc:
                review = OperatorReview(
                    verdict="not_reviewed",
                    notes=f"Operator review failed: {type(exc).__name__}: {exc}",
                    accepted_for_learning=False,
                )

            outcome = ExecutionOutcomeSchema(
                status=status,
                robot_success=robot_success,
                robot_message=robot_message,
                operator_verdict=review.verdict,
                operator_notes=review.notes,
                accepted_for_learning=review.accepted_for_learning,
            )
            record = TransitionRecordSchema(
                transition_id=transition_id,
                trace_id=self.trace_id,
                transition_index=transition_index,
                started_at_s=started_at_s,
                finished_at_s=finished_at_s,
                before_state_path=final_relative_directory / "before.yaml",
                action=prepared.action,
                outcome=outcome,
                after_state_path=final_relative_directory / "after.yaml",
            )

            stage = "writing transition record"
            record_staging_path = staging_directory / "transition.yaml"
            self._schema_writer(record, record_staging_path)

            # Reload all staged schemas before publishing the directory.  This catches truncated or
            # incorrectly encoded files while the physical attempt is still clearly marked.
            stage = "validating staged files"
            load_schema_yaml(StateSnapshotSchema, before_staging_path)
            load_schema_yaml(StateSnapshotSchema, after_staging_path)
            load_schema_yaml(TransitionRecordSchema, record_staging_path)

            stage = "committing transition"
            if final_directory.exists():
                raise FileExistsError(f"Refusing to overwrite {final_directory}.")
            staging_directory.rename(final_directory)
        except (Exception, KeyboardInterrupt) as exc:
            self._preserve_failed_attempt(
                staging_directory=staging_directory,
                transition_id=transition_id,
                transition_index=transition_index,
                stage=stage,
                error=exc,
                action=prepared.action,
            )
            if action_started:
                self._mark_trace_tainted(
                    transition_id=transition_id,
                    transition_index=transition_index,
                    stage=stage,
                    error=exc,
                )
            else:
                shutil.rmtree(staging_directory, ignore_errors=True)

            raise CollectionAttemptError(
                f"Could not commit transition '{transition_id}' while {stage}: {exc}",
                action_started=action_started,
                attempt_directory=staging_directory if staging_directory.exists() else None,
            ) from exc

        self._next_transition_index += 1

        interruption = execution_interrupt or review_interrupt
        if interruption is not None:
            raise interruption

        return record

    def _preserve_failed_attempt(
        self,
        *,
        staging_directory: Path,
        transition_id: str,
        transition_index: int,
        stage: str,
        error: BaseException,
        action: ExecutedSkillSchema,
    ) -> None:
        """Best-effort write diagnostic metadata alongside partial physical evidence."""
        if not staging_directory.exists():
            return

        error_data = {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "transition_id": transition_id,
            "transition_index": transition_index,
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "action": action.model_dump(mode="json"),
        }
        try:
            export_yaml_data(error_data, staging_directory / "collection_error.yaml")
        except Exception:
            # Never obscure the original collection error with a diagnostic-write failure.
            return

    def _mark_trace_tainted(
        self,
        *,
        transition_id: str,
        transition_index: int,
        stage: str,
        error: BaseException,
    ) -> None:
        """Prevent further collection after a possibly unrecorded physical state change."""
        self._tainted = True
        marker = {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "transition_id": transition_id,
            "transition_index": transition_index,
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "marked_at_s": time.time(),
            "recovery": "Physically reset the environment and begin a new trace.",
        }
        try:
            export_yaml_data(marker, self._trace_directory / self._TAINT_FILENAME)
        except Exception:
            # The in-memory flag still prevents this collector from proceeding.  A remaining hidden
            # staging directory prevents resume_trace from silently resuming after a restart.
            return
