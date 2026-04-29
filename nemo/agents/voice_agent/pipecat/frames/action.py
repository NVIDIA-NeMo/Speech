# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Action Frames.

The Action Frames implement the UMIM standard for frame processors.
See here to learn more about the UMIM standard:
https://docs.nvidia.com/ace/umim/latest/index.html

Note that at the moment the support for action frames is limited to the animation graph service.
More services and support for additional action frames will be added in the future.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from pipecat.frames.frames import ControlFrame, SystemFrame


def now_timestamp() -> datetime:
    """Helper to generate current timestamp."""
    return datetime.now(UTC)


def new_uid() -> str:
    """Helper to create a new UID."""
    return str(uuid4())


@dataclass
class ActionFrame:
    """Frame that belongs to an action with a defined start and end.

    Args:
        action_id (str): A unique id for the action.
        parent_action_ids (list[str]): The ids of parent actions (causing this action)

    """

    action_id: str = field(kw_only=True, default_factory=new_uid)
    parent_action_ids: list[str] = field(kw_only=True, default_factory=list)


@dataclass
class BotActionFrame(ActionFrame):
    """Frame related to a bot action.

    Args:
        bot_id (Optional[str]):  An ID identifying the bot performing the action. This field is required if you
            support multi-bot interactions.
    """

    bot_id: str | None = field(kw_only=True, default=None)


@dataclass
class UserActionFrame(ActionFrame):
    """Frame related to a user action.

    Args:
        user_id (Optional[str]):  An ID identifying the user performing the action. This field is required if you
            support multi-user interactions.
    """

    user_id: str | None = field(kw_only=True, default=None)


@dataclass
class StartActionFrame(ControlFrame, ActionFrame):
    """Event to start an action.

    All other actions that can be started inherit from this base spec.
    The action_id is used to differentiate between multiple runs of the same action.
    """


@dataclass
class StartedActionFrame(ControlFrame, ActionFrame):
    """The execution of an action has started.

    Args:
        action_started_at (datetime): The timestamp of when the action has started.

    """

    action_id: str = field(kw_only=True)
    action_started_at: datetime = field(kw_only=True, default_factory=now_timestamp)


@dataclass
class StopActionFrame(ControlFrame, ActionFrame):
    """An action needs to be stopped.

    This should be used to proactively stop an action that can take a
    longer period of time, e.g., a gesture.
    """

    action_id: str = field(kw_only=True)


@dataclass
class ChangeActionFrame(ControlFrame, ActionFrame):
    """The parameters of a running action needs to be changed.

    Updating running actions is useful for longer running
    actions (e.g. an avatar animation) which can adapt their behavior dynamically. For example, a nodding animation
    can change its speed depending on the voice activity level.
    """

    action_id: str = field(kw_only=True)


@dataclass
class UpdatedActionFrame(ControlFrame, ActionFrame):
    """A running action provides a (partial) result.

    Ongoing actions can provide partial updates on the current status
    of the action. An ActionUpdated should always update the payload of the action object and provide
    the type of update.

    Args:
        action_updated_at (datetime): The timestamp of when the action was updated. The timestamp should represent
            the system time the action actually changed, not the timestamp of when the `Updated` event was created
            (for this, there is the `event_created_at` field).

    """

    action_updated_at: datetime = field(kw_only=True, default_factory=now_timestamp)


@dataclass
class FinishedActionFrame(ControlFrame, ActionFrame):
    """An action has finished its execution.

    An action can finish either because the action has completed or
    failed (natural completion) or it can finish because it was stopped by the IM. The success (or failure) of the
    execution is marked using the status_code attribute.

    Args:
        action_finished_at (datetime): The timestamp of when the action has finished.
        is_success (bool): Did the action finish successfully
        was_stopped (Optional[bool]): Was the action stopped by a Stop event
        failure_reason (Optional[str]): Reason for action failure in case the action did not execute successfully
    """

    action_id: str = field(kw_only=True)
    action_finished_at: datetime = field(kw_only=True, default_factory=now_timestamp)
    is_success: bool = field(kw_only=True, default=True)
    was_stopped: bool | None = field(kw_only=True, default=None)
    failure_reason: str | None = field(kw_only=True, default=None)


# Presence User Action
@dataclass
class StartedPresenceUserActionFrame(StartedActionFrame, UserActionFrame, SystemFrame):
    """The interactive system detects the presence of a user in the system.

    TODO: We inherit from SystemFrame to circumvent the frame deletion issue with InterruptionFrame.
    This is a temporary fix only and needs to be reconsidered once the action concept is properly
    introduced.
    """


@dataclass
class FinishedPresenceUserActionFrame(FinishedActionFrame, UserActionFrame, SystemFrame):
    """The interactive system detects the user's absence.

    TODO: We inherit from SystemFrame to circumvent the frame deletion issue with InterruptionFrame.
    This is a temporary fix only and needs to be reconsidered once the action concept is properly
    introduced.
    """
