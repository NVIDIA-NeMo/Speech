# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA Context Aggregator.

This module provides specialized frame processors and context aggregators for
handling NVIDIA's Speculative Speech Processing feature in conversational AI systems.
It manages the processing and aggregation of interim and final transcripts,
enabling real-time response generation while maintaining conversation coherence.

The processors handle:
- Interim transcript processing for early response generation
- Context management for speculative responses
- TTS response caching and timing control for natural turn-taking
- Bidirectional conversation state management

Also see:
    pipecat.processors.aggregators.llm_response
    pipecat.processors.aggregators.llm_context

Classes:
    NvidiaAssistantContextAggregator: Handles assistant-specific context aggregation.
    NvidiaUserContextAggregator: Manages user context with interim/final transcripts.
    NvidiaTTSResponseCacher: Controls TTS response timing.
    NvidiaContextAggregatorPair: Coordinates paired aggregators.

Functions:
    create_nvidia_context_aggregator: Factory for creating aggregator pairs.
"""

from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nemo.agents.voice_agent.pipecat.frames.action import StartedPresenceUserActionFrame
from nemo.agents.voice_agent.pipecat.frames.riva import RivaInterimTranscriptionFrame


class NvidiaAssistantContextAggregator(LLMAssistantContextAggregator):
    """Extends LLMAssistantContextAggregator for NVIDIA-specific requirements.

    Specializes the base aggregator for handling speculative speech processing,
    managing assistant responses and context updates based on interim/final transcripts.

    Args:
        context (LLMContext): The context object to use.
        expect_stripped_words (bool): Whether to expect preprocessed words. Defaults to True.
        **kwargs: Additional arguments passed to parent class.

    Input Frames:
        LLMFullResponseStartFrame: Marks response start
        LLMFullResponseEndFrame: Marks response end
        TextFrame: Contains response text
        InterruptionFrame: Signals interruption

    Output Frames:
        LLMContextFrame: Updated context with responses
    """

    async def push_aggregation(self):
        """Updates the context with current aggregation.

        For speculative processing, may update existing messages rather than append
        to maintain context coherence with interim transcripts.
        - If the last message in context has the same role, it updates that message
        - Otherwise, appends a new message with the current aggregation
        - After pushing, resets the aggregation state

        Returns:
            None

        Typical usage example:
            >>> context = LLMContext()
            >>> aggregator = NvidiaAssistantContextAggregator(context)
            >>> # Update existing response
            >>> context.add_message({"role": "assistant", "content": "initial response"})
            >>> aggregator._aggregation = "updated response"
            >>> await aggregator.push_aggregation()
        """
        if len(self._aggregation) > 0:
            context_messages = self.context.get_messages()
            # Update existing message if same role, otherwise append new one
            if len(context_messages) > 0 and context_messages[-1]["role"] == self._role:
                context_messages[-1]["content"] = self._aggregation
                self.context.set_messages(context_messages)
            else:
                self.context.add_message({"role": self._role, "content": self._aggregation})
            self._aggregation = ""
            frame = LLMContextFrame(self.context)
            await self.push_frame(frame)
            # Reset our accumulator state.
            await self.reset()


class NvidiaUserContextAggregator(LLMUserContextAggregator):
    """Extends LLMUserContextAggregator for user-specific context handling.

    Handles speculative speech processing with interim and final transcriptions.
    Key features for speculative processing:
    - Processes stable interim transcripts for early response generation
    - Manages transition from interim to final transcripts
    - Deduplicates repeated transcripts to prevent context pollution
    - Maintains conversation history with configurable turn limits
    - Tracks user speaking state to coordinate with assistant responses

    Input Frames:
        TranscriptionFrame: Final transcription
        RivaInterimTranscriptionFrame: Interim transcription
        UserStartedSpeakingFrame: User began speaking
        UserStoppedSpeakingFrame: User stopped speaking
        InterruptionFrame: Conversation interruption

    Output Frames:
        LLMContextFrame: Updated context with transcripts
    """

    def __init__(
        self,
        send_interims: bool = True,
        chat_history_limit: int = 20,
        preserve_prompt_messages: int = 1,
        **kwargs,
    ):
        """Initialize the NvidiaUserContextAggregator.

        Args:
            send_interims (bool, optional): Whether to send interim transcription frames. Defaults to True.
            chat_history_limit (int): Limits the number of turns in chat history,
            preserve_prompt_messages (int): Number of initial prompt messages to always preserve.
                Defaults to 1. Set to 2 for Nemotron models to preserve system and first user message.
            **kwargs: Additional keyword arguments passed to parent LLMUserContextAggregator.
        """
        super().__init__(**kwargs)
        self.send_interims = send_interims
        self.chat_history_limit = chat_history_limit
        self.preserve_prompt_messages = preserve_prompt_messages
        self.last_transcript = None
        self._user_speaking = False
        self.seen_final = True
        self._last_final_transcript = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame for speculative speech handling.

        - Processes stable interim transcripts when user is speaking
        - Manages transition between interim and final transcripts
        - Handles user speaking state changes
        - Deduplicates repeated transcripts

        Args:
            frame: Frame to process (TranscriptionFrame, RivaInterimTranscriptionFrame,
                or user state frames)
            direction: Direction of frame flow in the pipeline

        Typical usage example:
            >>> aggregator = NvidiaUserContextAggregator(
            ...     send_interims=True,  # Enable interim transcript processing
            ...     chat_history_limit=20  # Keep last 20 conversation turns
            ... )
            >>> # Process final transcript
            >>> frame = TranscriptionFrame(text="Hello")
            >>> await aggregator.process_frame(frame, FrameDirection.DOWNSTREAM)
            >>>
            >>> # Process interim transcript
            >>> frame = RivaInterimTranscriptionFrame(text="Hello", stability=1.0)
            >>> await aggregator.process_frame(frame, FrameDirection.DOWNSTREAM)
        """
        if isinstance(frame, TranscriptionFrame):
            logger.debug(f"Recieved final transcript at NvidiaUserContextAggregator {frame.text}")
            # Only process if this is a new transcript
            if self.last_transcript is None or (self.last_transcript.rstrip() != frame.text.rstrip()):
                logger.debug(f"Sent final transcript downstream to LLM from NvidiaUserContextAggregator {frame.text}")
                self._aggregation = frame.text
                await self.push_aggregation()
                # Accumulate finals with proper spacing
                if self._last_final_transcript:
                    self._last_final_transcript += " " + frame.text
                else:
                    self._last_final_transcript = frame.text
            self.last_transcript = None
            self.seen_final = True
        elif isinstance(frame, RivaInterimTranscriptionFrame):
            # Process stable interim transcriptions during active speech and before first final result
            if (
                self.send_interims
                and (self._user_speaking or not self.seen_final)
                and abs(frame.stability - 1.0) < 1e-9
            ):
                logger.debug(
                    f"Sent interim transcript downstream to LLM from NvidiaUserContextAggregator {frame.text}"
                )
                self._aggregation = frame.text
                await self.push_aggregation()
                self.last_transcript = frame.text
        elif isinstance(frame, InterruptionFrame):
            self._user_speaking = False
            await self._start_interruption()
            await self.stop_all_metrics()
            await self.push_frame(frame, direction)
        else:
            if isinstance(frame, UserStartedSpeakingFrame):
                self._user_speaking = True
                self.seen_final = False
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._user_speaking = False
            await super().process_frame(frame, direction)

    async def get_truncated_context(self) -> LLMContext:
        """Returns a truncated context limited to specified chat history size.

        - Preserves initial messages (system + first user message) for Nemotron models
        - Counts conversation turns based on user-assistant exchanges
        - Preserves system and function messages regardless of limit
        - Processes messages in reverse order to maintain recent history

        Returns:
            LLMContext: New context object containing truncated conversation
                history, preserving system/function messages and most recent turns.

        Typical usage example:
            >>> aggregator = NvidiaUserContextAggregator(chat_history_limit=2)
            >>> # Context with 3 turns
            >>> context = LLMContext()
            >>> # Turn 1
            >>> context.add_message({"role": "user", "content": "Turn 1 user"})
            >>> context.add_message({"role": "assistant", "content": "Turn 1 assistant"})
            >>> # Turn 2
            >>> context.add_message({"role": "user", "content": "Turn 2 user"})
            >>> context.add_message({"role": "assistant", "content": "Turn 2 assistant"})
            >>> # Turn 3
            >>> context.add_message({"role": "user", "content": "Turn 3 user"})
            >>> context.add_message({"role": "assistant", "content": "Turn 3 assistant"})
            >>> # Get truncated context - will only contain most recent 2 turns
            >>> truncated = await aggregator.get_truncated_context()
            >>> print(truncated.get_messages())  # Shows turns 2 and 3 only
        """
        truncated_context = self.context
        if len(self.context.get_messages()) > 0:
            truncated_context = LLMContext()
            all_messages = self.context.get_messages()

            # Separate initial prompt messages to preserve from the rest
            initial_messages = all_messages[: self.preserve_prompt_messages]
            remaining_messages = all_messages[self.preserve_prompt_messages :]

            # Process remaining messages in reverse order to maintain recent history
            truncated_context_messages = []
            current_size = 0
            for context_message in reversed(remaining_messages):
                if (
                    context_message["role"] == "user"
                    or context_message["role"] == "assistant"
                    or context_message["role"] == "developer"
                    or context_message["role"] == "function"
                    or context_message["role"] == "tool"
                ):
                    if current_size == self.chat_history_limit:
                        continue
                    if context_message["role"] == "user":
                        current_size = current_size + 1
                truncated_context_messages.append(context_message)

            # Combine: initial messages + truncated recent messages (in correct order)
            final_messages = initial_messages + list(reversed(truncated_context_messages))
            truncated_context.set_messages(final_messages)
        return truncated_context

    async def push_aggregation(self):
        """Pushes aggregation to context and manages conversation flow.

        Updates or appends current aggregation to conversation context while
        maintaining turn-taking structure. For speculative responses, may update
        existing message rather than append new one.
        - If the last message in context has the same role, it updates that message
        - Otherwise, appends a new message with the current aggregation
        - After pushing, resets the aggregation state

        Output Frames:
            LLMContextFrame: downstream after processing.

        Typical usage example:
            >>> context = LLMContext()
            >>> aggregator = NvidiaUserContextAggregator(context)
            >>> # Update existing response
            >>> context.add_message({"role": "user", "content": "initial query"})
            >>> aggregator._aggregation = "updated query"
            >>> await aggregator.push_aggregation()
        """
        if len(self._aggregation) > 0:
            context_messages = self.context.get_messages()
            # Update existing message if same role, otherwise append new one
            if len(context_messages) > 0 and context_messages[-1]["role"] == self._role:
                if self._last_final_transcript:
                    context_messages[-1]["content"] = self._last_final_transcript + " " + self._aggregation
                else:
                    context_messages[-1]["content"] = self._aggregation
                self.context.set_messages(context_messages)
            else:
                self._last_final_transcript = ""
                self.context.add_message({"role": self._role, "content": self._aggregation})

            self._aggregation = ""
            # Get truncated context and send downstream
            truncated_context = await self.get_truncated_context()
            frame = LLMContextFrame(truncated_context)
            # Send the interruption before the context frame
            await self.push_frame(InterruptionFrame())
            logger.debug(
                f"Sending context downstream to LLM from NvidiaUserContextAggregator {frame.context.get_messages()}"
            )
            await self.push_frame(frame)
            # Reset our accumulator state
            await self.reset()


class NvidiaTTSResponseCacher(FrameProcessor):
    """Caches TTS responses and controls release timing for speculative speech.

    Manages text-to-speech response timing by caching responses and controlling
    their release based on user speaking state. Maintains natural turn-taking
    and prevents response overlap during speculative processing.

    Input frames handled:
        - LLMFullResponseStartFrame: Marks response start
        - LLMFullResponseEndFrame: Marks response end
        - TTSAudioRawFrame: TTS audio data
        - TTSStartedFrame: TTS start marker
        - TTSStoppedFrame: TTS stop marker
        - TTSTextFrame: TTS text data
        - UserStartedSpeakingFrame: Triggers caching
        - UserStoppedSpeakingFrame: Triggers release
        - InterruptionFrame: Clears cache
    """

    def __init__(self) -> None:
        """Initialize the NvidiaTTSResponseCacher."""
        super().__init__()
        self._cache: list[Frame] = []
        self.user_stopped_speaking: bool = True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes frame for TTS response caching and timing control.

        - Caches TTS responses while user is speaking
        - Releases cached responses when user stops speaking
        - Clears cache on interruptions
        - Maintains conversation flow by coordinating response timing

        Also see:
        - NvidiaUserContextAggregator : Handles user context and speech state
        - NvidiaAssistantContextAggregator : Manages assistant responses

        Args:
            frame: Frame to process
            direction: Direction of frame flow in pipeline

        Typical usage example:
            >>> cacher = NvidiaTTSResponseCacher()
            >>> # User starts speaking
            >>> await cacher.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            >>> # TTS response arrives - will be cached
            >>> await cacher.process_frame(TTSAudioRawFrame(audio_data), FrameDirection.DOWNSTREAM)
            >>> # User stops speaking - cached responses will be released
            >>> await cacher.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        """
        await super().process_frame(frame, direction)

        # Handle response start - cache if user is speaking
        if isinstance(frame, LLMFullResponseStartFrame):
            if self.user_stopped_speaking:
                await self.push_frame(frame, direction)
            else:
                self._cache.clear()  # Clear existing cache before new response
                self._cache.append(frame)

        # Handle TTS frames - cache or forward based on user speaking state
        elif isinstance(frame, (TTSAudioRawFrame | TTSStartedFrame | TTSStoppedFrame | TTSTextFrame)):
            if self.user_stopped_speaking:
                await self.push_frame(frame, direction)
            else:
                self._cache.append(frame)

        # Handle response end - mark user can speak after forwarding
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self.user_stopped_speaking:
                await self.push_frame(frame, direction)
                self.user_stopped_speaking = False  # Allow user to speak after response ends
            self._cache.append(frame)

        # Handle interruptions - clear cache and reset state
        elif isinstance(frame, InterruptionFrame | StartedPresenceUserActionFrame):
            # TODO: This only works if we have a single user in the system.
            # it also does not work if other "events" should trigger the cache release
            # (e.g. new frames by new processors).
            self._cache.clear()
            await self.push_frame(frame, direction)

        # Handle user stop speaking - release cached responses
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_stopped_speaking = True
            if self._cache:
                for cached_frame in self._cache:
                    await self.push_frame(cached_frame)
            self._cache.clear()
            await self.push_frame(frame, direction)

        # Handle user start speaking - update state
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.user_stopped_speaking = False
            await self.push_frame(frame, direction)

        # Forward all other frames unchanged
        else:
            await self.push_frame(frame, direction)


@dataclass
class NvidiaContextAggregatorPair:
    """A pair of context aggregators for managing bidirectional conversation.

    Attributes:
        _user: NvidiaUserContextAggregator for user-side context
        _assistant: NvidiaAssistantContextAggregator for assistant-side context
    """

    _user: "NvidiaUserContextAggregator"
    _assistant: "NvidiaAssistantContextAggregator"

    def user(self) -> "NvidiaUserContextAggregator":
        """Get the user context aggregator."""
        return self._user

    def assistant(self) -> "NvidiaAssistantContextAggregator":
        """Get the assistant context aggregator."""
        return self._assistant


def create_nvidia_context_aggregator(
    context: LLMContext,
    assistant_expect_stripped_words: bool = True,
    send_interims: bool = False,
    chat_history_limit: int = 20,
    preserve_prompt_messages: int = 1,
) -> NvidiaContextAggregatorPair:
    """Creates a pair of context aggregators for speculative speech processing.

    - Creates synchronized user and assistant aggregators sharing context
    - User aggregator handles interim/final transcripts
    - Assistant aggregator manages response generation
    - Both work together to maintain conversation coherence

    Also see:
    - NvidiaUserContextAggregator : Handles user context
    - NvidiaAssistantContextAggregator : Handles assistant context

    Args:
        context: Base context object to initialize aggregators
        assistant_expect_stripped_words: Whether assistant expects preprocessed words
        send_interims: Whether to process interim transcriptions
        chat_history_limit: Maximum number of conversation turns to maintain
        preserve_prompt_messages: Number of initial prompt messages to always preserve.
            Defaults to 1. Set to 2 for Nemotron models to preserve system and first user message.

    Returns:
    NvidiaContextAggregatorPair: A paired set of user and assistant context aggregators configured for
    speculative speech processing.

    Typical usage example:
        >>> context = LLMContext()
        >>> # Create aggregators with default settings
        >>> aggregators = create_nvidia_context_aggregator(context)
        >>>
        >>> # Create aggregators with custom settings
        >>> aggregators = create_nvidia_context_aggregator(
        ...     context,
        ...     send_interims=True,  # Enable interim transcript processing
        ...     chat_history_limit=10,  # Keep shorter history
        ...     preserve_prompt_messages=2,  # For Nemotron: preserve system + first user
        ...     assistant_expect_stripped_words=False  # Raw word processing
        ... )
        >>>
        >>> # Access individual aggregators
        >>> user_aggregator = aggregators.user()
        >>> assistant_aggregator = aggregators.assistant()
    """
    # Create user aggregator with specified settings
    user_params = LLMUserAggregatorParams(aggregation_timeout=0.01)
    user = NvidiaUserContextAggregator(
        send_interims=send_interims,
        context=context,
        params=user_params,
        chat_history_limit=chat_history_limit,
        preserve_prompt_messages=preserve_prompt_messages,
    )
    # Create assistant aggregator sharing context with user
    assistant_params = LLMAssistantAggregatorParams(expect_stripped_words=assistant_expect_stripped_words)
    assistant = NvidiaAssistantContextAggregator(context=user.context, params=assistant_params)
    return NvidiaContextAggregatorPair(_user=user, _assistant=assistant)
