"""Draft, versioned data contract for Spot transition collection.

This module is intentionally independent of ROS, ``SpotSkillsProtocol``, and the
abstraction-learning implementation.  It defines only the data that a collector should save.
Keeping the contract separate makes it possible to review and revise the on-disk format before
connecting it to the physical robot.

All paths stored by these models are relative to the dataset root.  All poses include an explicit
reference frame.  Large observations (images and occupancy grids) are stored in separate files and
referenced by path instead of being embedded in transition YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Annotated

from robotics_utils.io.pydantic_schemata import Pose3DDictSchema
from robotics_utils.io.yaml_utils import export_yaml_data, load_yaml_data


SCHEMA_VERSION = 1
"""Current version of the Spot collection file format."""


def _validate_dataset_relative_path(path: Path) -> Path:
    """Require a portable path beneath the dataset root."""
    if path.is_absolute():
        raise ValueError(f"Dataset paths must be relative, not absolute: '{path}'.")
    if ".." in path.parts:
        raise ValueError(f"Dataset paths cannot escape the dataset root: '{path}'.")
    return path


class ObservedPoseSchema(BaseModel):
    """A framed pose together with its measurement provenance."""

    pose: Pose3DDictSchema
    source: Literal["configured", "fiducial", "tf", "kinematic", "manual"]
    observed_at_s: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class SnapshotObjectSchema(BaseModel):
    """Dynamic perceptual state of one world object."""

    name: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    pose: Optional[ObservedPoseSchema] = None

    # This field is present only for objects that can be opened and closed.  It is intended to
    # represent a directly observed or manually verified configuration, not a learned predicate.
    container_status: Optional[Literal["open", "closed"]] = None

    # Each path points to an ImageObservation YAML file, which in turn references an image file and
    # records the camera pose used to obtain it.
    image_observations: List[Path] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("image_observations")
    @classmethod
    def validate_image_observation_paths(cls, paths: List[Path]) -> List[Path]:
        """Require all image-observation paths to be dataset-relative."""
        return [_validate_dataset_relative_path(path) for path in paths]


class SnapshotRobotSchema(BaseModel):
    """Dynamic state of one robot."""

    name: str = Field(min_length=1)
    base_pose: ObservedPoseSchema

    # Joint names map to positions in radians or meters, following the corresponding joint model.
    configuration: Dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraspSnapshotSchema(BaseModel):
    """A grasp attachment reported by the robot's kinematic state."""

    robot_name: str = Field(min_length=1)
    end_effector_name: str = Field(min_length=1)
    object_name: str = Field(min_length=1)
    pose_ee_object: Pose3DDictSchema

    model_config = ConfigDict(extra="forbid", frozen=True)


class StateSnapshotSchema(BaseModel):
    """Versioned, immutable observation of the Spot environment at one instant."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    snapshot_id: str = Field(min_length=1)
    captured_at_s: float = Field(ge=0)
    root_frame: str = Field(min_length=1)

    # Identifies the static environment/collision configuration used for this observation.
    environment_config: Path

    robots: List[SnapshotRobotSchema]
    objects: List[SnapshotObjectSchema]
    grasps: List[GraspSnapshotSchema] = Field(default_factory=list)

    # Large map data remains in its native file format and is referenced when it changes.
    occupancy_grid_path: Optional[Path] = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("environment_config")
    @classmethod
    def validate_environment_config_path(cls, path: Path) -> Path:
        """Require the environment configuration path to be dataset-relative."""
        return _validate_dataset_relative_path(path)

    @field_validator("occupancy_grid_path")
    @classmethod
    def validate_occupancy_grid_path(cls, path: Optional[Path]) -> Optional[Path]:
        """Require the optional occupancy-grid path to be dataset-relative."""
        return None if path is None else _validate_dataset_relative_path(path)

    @model_validator(mode="after")
    def validate_names_and_grasp_references(self) -> StateSnapshotSchema:
        """Require unique entities and internally consistent grasp references."""
        robot_names = [robot.name for robot in self.robots]
        object_names = [obj.name for obj in self.objects]

        if len(robot_names) != len(set(robot_names)):
            raise ValueError("Robot names must be unique within a state snapshot.")
        if len(object_names) != len(set(object_names)):
            raise ValueError("Object names must be unique within a state snapshot.")

        known_robots = set(robot_names)
        known_objects = set(object_names)
        grasped_objects: Set[str] = set()
        occupied_end_effectors: Set[Tuple[str, str]] = set()

        for grasp in self.grasps:
            if grasp.robot_name not in known_robots:
                raise ValueError(f"Grasp references unknown robot '{grasp.robot_name}'.")
            if grasp.object_name not in known_objects:
                raise ValueError(f"Grasp references unknown object '{grasp.object_name}'.")
            if grasp.object_name in grasped_objects:
                raise ValueError(f"Object '{grasp.object_name}' has more than one grasp.")

            end_effector = (grasp.robot_name, grasp.end_effector_name)
            if end_effector in occupied_end_effectors:
                raise ValueError(
                    f"End-effector '{grasp.robot_name}/{grasp.end_effector_name}' has more than "
                    "one grasp.",
                )

            grasped_objects.add(grasp.object_name)
            occupied_end_effectors.add(end_effector)

        return self


# Skill arguments use a tagged union.  The tag is necessary because object names, waypoint names,
# paths, and arbitrary strings are all represented by Python strings at execution time but have
# different meanings to the abstraction-learning pipeline.
class ObjectArgumentSchema(BaseModel):
    """A binding to a named object in the environment."""

    name: str = Field(min_length=1)
    kind: Literal["object"]
    value: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class WaypointArgumentSchema(BaseModel):
    """A binding to a named navigation waypoint."""

    name: str = Field(min_length=1)
    kind: Literal["waypoint"]
    value: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class PoseArgumentSchema(BaseModel):
    """A binding to a framed continuous pose."""

    name: str = Field(min_length=1)
    kind: Literal["pose"]
    value: Pose3DDictSchema

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContinuousArgumentSchema(BaseModel):
    """A binding to a scalar continuous execution parameter."""

    name: str = Field(min_length=1)
    kind: Literal["continuous"]
    value: float

    model_config = ConfigDict(extra="forbid", frozen=True)


class IntegerArgumentSchema(BaseModel):
    """A binding to an integer execution parameter."""

    name: str = Field(min_length=1)
    kind: Literal["integer"]
    value: int

    model_config = ConfigDict(extra="forbid", frozen=True)


class BooleanArgumentSchema(BaseModel):
    """A binding to a Boolean execution parameter."""

    name: str = Field(min_length=1)
    kind: Literal["boolean"]
    value: bool

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathArgumentSchema(BaseModel):
    """A binding to a portable dataset/configuration path."""

    name: str = Field(min_length=1)
    kind: Literal["path"]
    value: Path

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("value")
    @classmethod
    def validate_path(cls, path: Path) -> Path:
        """Require the argument path to be dataset-relative."""
        return _validate_dataset_relative_path(path)


class StringArgumentSchema(BaseModel):
    """A binding to a string with no object or waypoint semantics."""

    name: str = Field(min_length=1)
    kind: Literal["string"]
    value: str

    model_config = ConfigDict(extra="forbid", frozen=True)


SkillArgumentSchema = Annotated[
    Union[
        ObjectArgumentSchema,
        WaypointArgumentSchema,
        PoseArgumentSchema,
        ContinuousArgumentSchema,
        IntegerArgumentSchema,
        BooleanArgumentSchema,
        PathArgumentSchema,
        StringArgumentSchema,
    ],
    Field(discriminator="kind"),
]
"""Serializable argument types supported by a collected skill invocation."""


class ExecutedSkillSchema(BaseModel):
    """A complete, serializable, grounded skill invocation."""

    name: str = Field(min_length=1)
    arguments: List[SkillArgumentSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_unique_argument_names(self) -> ExecutedSkillSchema:
        """Require each formal parameter to have exactly one recorded binding."""
        argument_names = [argument.name for argument in self.arguments]
        if len(argument_names) != len(set(argument_names)):
            raise ValueError("Skill argument names must be unique.")
        return self


class ExecutionOutcomeSchema(BaseModel):
    """Separate assessments made by the robot software and the human operator."""

    status: Literal["completed", "exception", "interrupted"]
    robot_success: Optional[bool]
    robot_message: str = ""

    operator_verdict: Literal["success", "failure", "uncertain", "not_reviewed"]
    operator_notes: Optional[str] = None

    # This flag controls dataset curation; it does not alter either success assessment above.
    accepted_for_learning: bool = True

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_completed_outcome(self) -> ExecutionOutcomeSchema:
        """Require a robot success result when execution returned normally."""
        if self.status == "completed" and self.robot_success is None:
            raise ValueError("A completed execution must include robot_success.")
        return self

    def resolved_success(self) -> Optional[bool]:
        """Resolve a learning label without discarding either original assessment.

        A decisive human verdict takes precedence.  Otherwise, the robot's report is returned.
        ``None`` means that neither source supplies a trustworthy binary label.
        """
        if self.operator_verdict == "success":
            return True
        if self.operator_verdict == "failure":
            return False
        if self.operator_verdict == "uncertain":
            return None
        return self.robot_success


class TransitionRecordSchema(BaseModel):
    """Durable record of one attempted physical skill execution."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    transition_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    transition_index: int = Field(ge=0)

    started_at_s: float = Field(ge=0)
    finished_at_s: float = Field(ge=0)

    before_state_path: Path
    action: ExecutedSkillSchema
    outcome: ExecutionOutcomeSchema
    after_state_path: Path

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("before_state_path", "after_state_path")
    @classmethod
    def validate_state_paths(cls, path: Path) -> Path:
        """Require both state references to be dataset-relative."""
        return _validate_dataset_relative_path(path)

    @model_validator(mode="after")
    def validate_transition(self) -> TransitionRecordSchema:
        """Validate temporal ordering and distinct state records."""
        if self.finished_at_s < self.started_at_s:
            raise ValueError("finished_at_s cannot precede started_at_s.")
        if self.before_state_path == self.after_state_path:
            raise ValueError("Before- and after-state snapshots must use different files.")
        return self


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def save_schema_yaml(schema: BaseModel, output_path: Path) -> None:
    """Serialize a collection schema to YAML using JSON-compatible primitive values."""
    export_yaml_data(schema.model_dump(mode="json"), output_path)


def load_schema_yaml(schema_type: Type[SchemaT], input_path: Path) -> SchemaT:
    """Load and validate a collection schema from YAML."""
    return schema_type.model_validate(load_yaml_data(input_path))
