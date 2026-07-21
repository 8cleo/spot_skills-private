"""Tests for ROS-independent Spot transition collection orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from robotics_utils.skills import Outcome, SkillsProtocol, skill_method
from spot_skills_py.spot_collection import (
    CollectionAttemptError,
    OperatorReview,
    ReviewRequest,
    SkillExecutionSpec,
    SnapshotCaptureRequest,
    SpotTransitionCollector,
    TraceNotUsableError,
)
from spot_skills_py.spot_collection_schemas import (
    ObservedPoseSchema,
    SnapshotObjectSchema,
    SnapshotRobotSchema,
    StateSnapshotSchema,
    TransitionRecordSchema,
    load_schema_yaml,
)


class FakeSkillsProtocol(SkillsProtocol):
    """Small skills protocol that exposes controllable physical-execution outcomes."""

    def __init__(
        self,
        events: list[str],
        *,
        success: bool = True,
        raise_on_execute: bool = False,
    ) -> None:
        """Initialize the fake execution behavior."""
        self.events = events
        self.success = success
        self.raise_on_execute = raise_on_execute
        self.execution_count = 0

    @skill_method
    def move(self, object_name: str = "eraser1", distance_m: float = 0.5) -> Outcome:
        """Move a named object by a configured distance.

        :param object_name: Name of the object moved by this skill
        :param distance_m: Distance in meters the object should move
        :return: Fake physical execution result
        """
        self.events.append("execute")
        self.execution_count += 1
        if self.raise_on_execute:
            raise RuntimeError("fake actuator error")
        return Outcome(success=self.success, message="fake outcome")


def _pose_observation() -> ObservedPoseSchema:
    """Construct a representative map-frame observation."""
    return ObservedPoseSchema(
        pose={"xyz_rpy": [0, 0, 0.56, 0, 0, 0], "frame": "map"},
        source="tf",
        observed_at_s=1.0,
    )


def _snapshot(request: SnapshotCaptureRequest) -> StateSnapshotSchema:
    """Construct a state whose ID matches the collector's capture request."""
    return StateSnapshotSchema(
        snapshot_id=request.snapshot_id,
        captured_at_s=1.0,
        root_frame="map",
        environment_config=Path("config/env.yaml"),
        robots=[SnapshotRobotSchema(name="spot", base_pose=_pose_observation())],
        objects=[SnapshotObjectSchema(name="eraser1", object_type="eraser")],
    )


def _review_success(events: list[str]):
    """Create an operator reviewer that records when it was called."""

    def review(_: ReviewRequest) -> OperatorReview:
        events.append("review")
        return OperatorReview(verdict="success", notes="looks good")

    return review


def _clock(values: list[float]):
    """Return a deterministic clock backed by the supplied sequence."""
    timestamps: Iterator[float] = iter(values)
    return timestamps.__next__


def _new_collector(
    tmp_path: Path,
    executor: FakeSkillsProtocol,
    capture_state,
    *,
    reviewer,
) -> SpotTransitionCollector:
    """Create a collector configured for the fake Move skill."""
    config_path = tmp_path / "config" / "env.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("default_frame: map\n")
    return SpotTransitionCollector.new_trace(
        executor=executor,
        capture_state=capture_state,
        dataset_root=tmp_path,
        trace_id="trace-001",
        allowed_skills=frozenset({"Move"}),
        argument_kinds={("Move", "object_name"): "object"},
        reviewer=reviewer,
        clock=_clock([10.0, 11.0]),
        id_factory=lambda: "transition-001",
    )


def test_collect_transition_orders_events_and_materializes_defaults(tmp_path: Path) -> None:
    """A committed action should contain fresh states and every resolved default argument."""
    # Arrange
    events: list[str] = []
    executor = FakeSkillsProtocol(events)

    def capture(request: SnapshotCaptureRequest) -> StateSnapshotSchema:
        events.append(f"capture-{request.phase}")
        return _snapshot(request)

    collector = _new_collector(
        tmp_path,
        executor,
        capture,
        reviewer=_review_success(events),
    )

    # Act
    record = collector.collect_transition(SkillExecutionSpec("Move"))

    # Assert
    assert events == ["capture-before", "execute", "capture-after", "review"]
    assert record.transition_index == 0
    assert record.started_at_s == 10.0
    assert record.finished_at_s == 11.0
    assert [argument.name for argument in record.action.arguments] == [
        "object_name",
        "distance_m",
    ]
    assert [argument.value for argument in record.action.arguments] == ["eraser1", 0.5]
    assert (tmp_path / record.before_state_path).is_file()
    assert (tmp_path / record.after_state_path).is_file()

    record_path = (tmp_path / record.before_state_path).parent / "transition.yaml"
    assert load_schema_yaml(TransitionRecordSchema, record_path) == record
    assert collector.next_transition_index == 1


def test_unsuccessful_outcome_still_commits_a_post_state(tmp_path: Path) -> None:
    """A physical skill failure is a useful negative transition, not a collection failure."""
    # Arrange
    events: list[str] = []
    executor = FakeSkillsProtocol(events, success=False)

    def capture(request: SnapshotCaptureRequest) -> StateSnapshotSchema:
        events.append(f"capture-{request.phase}")
        return _snapshot(request)

    collector = _new_collector(
        tmp_path,
        executor,
        capture,
        reviewer=lambda _: OperatorReview(verdict="failure"),
    )

    # Act
    record = collector.collect_transition(SkillExecutionSpec("Move"))

    # Assert
    assert record.outcome.status == "completed"
    assert record.outcome.robot_success is False
    assert record.outcome.operator_verdict == "failure"
    assert (tmp_path / record.after_state_path).is_file()


def test_execution_exception_still_commits_a_post_state(tmp_path: Path) -> None:
    """An exception after motion begins should be retained if the post-state is trustworthy."""
    # Arrange
    events: list[str] = []
    executor = FakeSkillsProtocol(events, raise_on_execute=True)

    def capture(request: SnapshotCaptureRequest) -> StateSnapshotSchema:
        events.append(f"capture-{request.phase}")
        return _snapshot(request)

    collector = _new_collector(
        tmp_path,
        executor,
        capture,
        reviewer=lambda _: OperatorReview(verdict="uncertain", accepted_for_learning=False),
    )

    # Act
    record = collector.collect_transition(SkillExecutionSpec("Move"))

    # Assert
    assert events == ["capture-before", "execute", "capture-after"]
    assert record.outcome.status == "exception"
    assert record.outcome.robot_success is None
    assert "RuntimeError: fake actuator error" in record.outcome.robot_message
    assert (tmp_path / record.after_state_path).is_file()


def test_pre_state_failure_does_not_execute_or_taint_trace(tmp_path: Path) -> None:
    """Failure before the action should be safely retryable at the same trace index."""
    # Arrange
    events: list[str] = []
    executor = FakeSkillsProtocol(events)

    def fail_capture(_: SnapshotCaptureRequest) -> StateSnapshotSchema:
        raise RuntimeError("camera unavailable")

    collector = _new_collector(
        tmp_path,
        executor,
        fail_capture,
        reviewer=lambda _: OperatorReview(verdict="not_reviewed"),
    )

    # Act/Assert
    with pytest.raises(CollectionAttemptError) as error:
        collector.collect_transition(SkillExecutionSpec("Move"))

    assert error.value.action_started is False
    assert executor.execution_count == 0
    assert collector.next_transition_index == 0
    assert not (tmp_path / "traces" / "trace-001" / "TRACE_TAINTED.yaml").exists()
    assert not list((tmp_path / "traces" / "trace-001" / "transitions").iterdir())


def test_missing_image_artifact_prevents_execution(tmp_path: Path) -> None:
    """A snapshot must not validate merely because its missing image path is well formed."""
    # Arrange
    events: list[str] = []
    executor = FakeSkillsProtocol(events)

    def capture(request: SnapshotCaptureRequest) -> StateSnapshotSchema:
        return StateSnapshotSchema(
            snapshot_id=request.snapshot_id,
            captured_at_s=1.0,
            root_frame="map",
            environment_config=Path("config/env.yaml"),
            robots=[SnapshotRobotSchema(name="spot", base_pose=_pose_observation())],
            objects=[
                SnapshotObjectSchema(
                    name="eraser1",
                    object_type="eraser",
                    image_observations=[request.asset_relative_directory / "missing.yaml"],
                ),
            ],
        )

    collector = _new_collector(
        tmp_path,
        executor,
        capture,
        reviewer=lambda _: OperatorReview(verdict="not_reviewed"),
    )

    # Act/Assert
    with pytest.raises(CollectionAttemptError, match="Image observation does not exist") as error:
        collector.collect_transition(SkillExecutionSpec("Move"))

    assert error.value.action_started is False
    assert executor.execution_count == 0


def test_post_state_failure_taints_trace_and_preserves_partial_evidence(tmp_path: Path) -> None:
    """An unrecorded endpoint must prevent another action from hiding the physical change."""
    # Arrange
    events: list[str] = []
    executor = FakeSkillsProtocol(events)

    def capture(request: SnapshotCaptureRequest) -> StateSnapshotSchema:
        events.append(f"capture-{request.phase}")
        if request.phase == "after":
            raise RuntimeError("post-state camera unavailable")
        return _snapshot(request)

    collector = _new_collector(
        tmp_path,
        executor,
        capture,
        reviewer=lambda _: OperatorReview(verdict="not_reviewed"),
    )

    # Act/Assert
    with pytest.raises(CollectionAttemptError) as error:
        collector.collect_transition(SkillExecutionSpec("Move"))

    assert error.value.action_started is True
    assert error.value.attempt_directory is not None
    assert (error.value.attempt_directory / "before.yaml").is_file()
    assert (error.value.attempt_directory / "collection_error.yaml").is_file()
    assert (tmp_path / "traces" / "trace-001" / "TRACE_TAINTED.yaml").is_file()

    with pytest.raises(TraceNotUsableError):
        collector.collect_transition(SkillExecutionSpec("Move"))
