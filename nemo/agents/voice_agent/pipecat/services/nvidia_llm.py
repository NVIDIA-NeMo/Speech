# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA LLM service implementation for interacting with NIM (NVIDIA Inference Microservice) API."""

from __future__ import annotations

import json
import time

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesUpdateFrame,
    LLMTextFrame,
    UserImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.tracing.service_decorators import traced_llm


class NvidiaLLMService(OpenAILLMService):
    """A service for interacting with NVIDIA's NIM (NVIDIA Inference Microservice) API.

    This service extends OpenAILLMService to work with NVIDIA's NIM API while maintaining
    compatibility with the OpenAI-style interface. It specifically handles the difference
    in token usage reporting between NIM (incremental) and OpenAI (final summary).

    Args:
        api_key (str): The API key for accessing NVIDIA's NIM API
        base_url (str, optional): The base URL for NIM API. Defaults to "https://integrate.api.nvidia.com/v1".
            For locally deployed NIM models, the corresponding endpoint can be passed in as a string.
        model (str, optional): The model identifier to use. Defaults to "meta/llama3-8b-instruct"
        filter_think_tokens (bool, optional): If True, filters out internal "thinking" tokens
            (content before the first </think> tag) from the LLM response. Only enable if your model produces
            thinking tokens with </think> tags. Defaults to False.
        mistral_model_support (bool, optional): If True, ensures that messages strictly alternate between user and
            assistant roles after the optional system prompt by combining consecutive messages from the same role.
            This is required for Mistral models. Defaults to False.
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: str = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "meta/llama3-8b-instruct",
        filter_think_tokens: bool = False,  # Only enable if model produces thinking tokens with </think> tags
        mistral_model_support: bool = False,  # Enable for Mistral models requiring user/assistant alternation
        text_aggregator: BaseTextAggregator | None = None,
        **kwargs,
    ):
        """Initialize the NvidiaLLMService with configuration parameters."""
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        # Counters for accumulating token usage metrics
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = False
        self._filter_think_tokens = filter_think_tokens
        self._mistral_model_support = mistral_model_support
        self._current_task = None
        self._text_aggregator = text_aggregator

        # State for think token filtering
        self._reset_think_filter_state()

        # State for first sentence generation timing
        self._first_sentence_detected = False
        self._first_sentence_start_time = None

    def _raise_llm_error(self, err: Exception) -> None:
        """Re-raise with a clear message when the error is due to endpoint/model config."""
        msg = str(err).lower()
        hint = "Check NVIDIA_LLM_URL, NVIDIA_LLM_MODEL, and NVIDIA_API_KEY."
        if isinstance(err, httpx.HTTPStatusError):
            if err.response.status_code == 404:
                raise ValueError(f"Nvidia LLM returned 404 (wrong URL or model). {hint}") from err
            if err.response.status_code == 401:
                raise ValueError(f"Nvidia LLM returned 401 (bad API key). {hint}") from err
            raise ValueError(f"Nvidia LLM returned HTTP {err.response.status_code}. {hint}") from err
        if isinstance(err, (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException)):
            raise ValueError(f"Cannot connect to Nvidia LLM. {hint}") from err
        if "404" in msg or "not found" in msg:
            raise ValueError(f"Nvidia LLM 404 or model not found. {hint}") from err
        if "401" in msg or "unauthorized" in msg:
            raise ValueError(f"Nvidia LLM 401 (bad API key). {hint}") from err
        if "connection" in msg or "connect" in msg or "refused" in msg:
            raise ValueError(f"Cannot connect to Nvidia LLM. {hint}") from err
        raise err

    def _reset_think_filter_state(self):
        """Reset the state variables used for think token filtering."""
        self.FULL_END_TAG = "</think>"
        self._seen_end_tag = False
        self._buffer = ""
        self._output_buffer = ""
        self._thinking_aggregation = ""
        self._partial_tag_buffer = ""

    def _preprocess_messages_for_mistral(
        self, messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """Preprocess messages for Mistral model compatibility by combining consecutive messages with the same role.

        This is required for Mistral models which expect strict alternation between user and assistant messages
        after an optional system message. This preprocessing combines consecutive messages with the same role
        into a single message.

        Args:
            messages (List[ChatCompletionMessageParam]): Original message list from the context

        Returns:
            List[ChatCompletionMessageParam]: Processed messages with consecutive same-role messages combined
        """
        if not self._mistral_model_support or len(messages) <= 1:
            return messages

        processed_messages = []
        current_role = None
        combined_content = ""

        # Loop through all messages and combine consecutive ones with the same role
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == current_role:
                # Same role as previous, combine content
                if content:
                    if combined_content:
                        combined_content += " " + content
                    else:
                        combined_content = content
            else:
                # New role, add the previous combined message if it exists
                if current_role is not None:
                    processed_messages.append({"role": current_role, "content": combined_content})

                # Start new combined message
                current_role = role
                combined_content = content

        # Add the last combined message
        if current_role is not None:
            processed_messages.append({"role": current_role, "content": combined_content})

        return processed_messages

    async def _process_context(self, context: LLMContext):
        """Process a context through the LLM and accumulate token usage metrics.

        This method overrides the parent class implementation to handle NVIDIA's
        incremental token reporting style, accumulating the counts and reporting
        them once at the end of processing. It also handles:

        1. Mistral model message preprocessing to combine consecutive messages with the same role
        2. Skipping LLM calls if only a system message is provided (Mistral models requirement)
        3. Duplicate function names and arguments that can occur with NVIDIA models
        4. Internal "thinking" token filtering if enabled


        Args:
            context (LLMContext): The context to process, containing messages
                and other information needed for the LLM interaction.
        """
        # Apply Mistral model preprocessing to ensure compatibility
        if self._mistral_model_support:
            original_messages = context.get_messages()
            if original_messages:
                processed_messages = self._preprocess_messages_for_mistral(original_messages)

                # Skip processing if the last (or only) message is a system message
                if processed_messages[-1].get("role") == "system":
                    logger.debug("Only system message is provided in the context, so skipping the LLM call.")
                    return

                context.set_messages(processed_messages)

        # Reset all counters and flags at the start of processing
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._has_reported_prompt_tokens = False
        self._is_processing = True

        # Reset think token filtering state
        if self._filter_think_tokens:
            self._reset_think_filter_state()

        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        try:
            await self.start_ttfb_metrics()
            chunk_stream = await self._stream_chat_completions_universal_context(context)

            async for chunk in chunk_stream:
                if chunk.usage:
                    tokens = LLMTokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                    await self.start_llm_usage_metrics(tokens)

                if chunk.choices is None or len(chunk.choices) == 0:
                    continue

                await self.stop_ttfb_metrics()

                if not chunk.choices[0].delta:
                    continue

                if chunk.choices[0].delta.tool_calls:
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if tool_call.index != func_idx:
                        functions_list.append(function_name)
                        arguments_list.append(arguments)
                        tool_id_list.append(tool_call_id)
                        function_name = ""
                        arguments = ""
                        tool_call_id = ""
                        func_idx += 1
                    if tool_call.function and tool_call.function.name:
                        # For locally deployed and nvdev models that send duplicate function names
                        if not function_name:
                            function_name = tool_call.function.name
                        elif tool_call.function.name != function_name:
                            # Only append if it's not a duplicate of the current complete name
                            function_name += tool_call.function.name
                        tool_call_id = tool_call.id
                    if tool_call.function and tool_call.function.arguments:
                        # Check for duplicate argument chunks (locally deployed and nvdev models issue)
                        if not arguments:
                            arguments = tool_call.function.arguments
                        elif tool_call.function.arguments not in arguments:
                            # Only append if this chunk is not already in the arguments
                            arguments += tool_call.function.arguments
                elif chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content

                    # Filter think tokens if enabled
                    if self._filter_think_tokens:
                        filtered_content = self._filter_think_token(content)
                        await self.push_frame(LLMTextFrame(filtered_content))
                    else:
                        await self.push_frame(LLMTextFrame(content))

                        # Check for first sentence completion using configured aggregator or default matcher
                        if content and not self._first_sentence_detected:
                            if self._text_aggregator is None:
                                end_of_sentence_pos = match_endofsentence(content)
                                if end_of_sentence_pos > 0:
                                    self._first_sentence_detected = True

                            if self._first_sentence_detected and self._first_sentence_start_time is not None:
                                first_sentence_time = time.time() - self._first_sentence_start_time
                                logger.debug(f"{self} LLM first sentence generation time: {first_sentence_time:.3f}")

            # Process any remaining content in buffers
            if self._filter_think_tokens and not self._seen_end_tag and self._thinking_aggregation:
                # No </think> tag was ever seen even after enabling filtering thinking tokens,
                # so treat everything as actual response and push the aggregated content
                await self.push_frame(LLMTextFrame(self._thinking_aggregation))
                self._reset_think_filter_state()

            # if we got a function name and arguments, check to see if it's a function with
            # a registered handler. If so, run the registered callback, save the result to
            # the context, and re-prompt to get a chat answer. If we don't have a registered
            # handler, raise an exception.
            if function_name and arguments:
                # added to the list as last function name and arguments not added to the list
                functions_list.append(function_name)
                arguments_list.append(arguments)
                tool_id_list.append(tool_call_id)

                for _index, (function_name, arguments, tool_id) in enumerate(
                    zip(functions_list, arguments_list, tool_id_list, strict=False), start=1
                ):
                    if await self.has_function(function_name):
                        run_llm = False
                        arguments = json.loads(arguments)
                        await self.call_function(
                            context=context,
                            function_name=function_name,
                            arguments=arguments,
                            tool_call_id=tool_id,
                            run_llm=run_llm,
                        )
                    else:
                        raise Exception(
                            f"The LLM tried to call a function named '{function_name}', "
                            f"but there isn't a callback registered for that function."
                        )
        except Exception as e:
            self._raise_llm_error(e)
        finally:
            self._is_processing = False
            # Report final accumulated token usage at the end of processing
            if self._prompt_tokens > 0 or self._completion_tokens > 0:
                self._total_tokens = self._prompt_tokens + self._completion_tokens
                tokens = LLMTokenUsage(
                    prompt_tokens=self._prompt_tokens,
                    completion_tokens=self._completion_tokens,
                    total_tokens=self._total_tokens,
                )
                await super().start_llm_usage_metrics(tokens)

    def _filter_think_token(self, content: str) -> str:
        """Filter content by ignoring everything before the first </think> tag.

        After the first </think>, all content is considered actual response.
        If no </think> tag is found, the entire response is treated as actual output at the end of the call.
        Handles cases where the </think> tag might be split across multiple streaming tokens.
        """
        if self._seen_end_tag:
            return content  # Already past the think, just return content

        # Add new content to buffer
        self._buffer += content
        self._thinking_aggregation += content
        filtered_content = ""

        # Check if we have a complete tag in the buffer
        if self.FULL_END_TAG in self._buffer:
            end_tag_idx = self._buffer.find(self.FULL_END_TAG)

            # Found the end tag, everything after it is real content
            self._seen_end_tag = True
            after_tag = self._buffer[end_tag_idx + len(self.FULL_END_TAG) :]
            filtered_content = after_tag

            # Clear buffers
            self._buffer = ""
            self._thinking_aggregation = ""
            self._partial_tag_buffer = ""
            return filtered_content

        # Check for partial tag at the end of buffer
        end_chars = min(len(self._buffer), len(self.FULL_END_TAG) - 1)
        for i in range(1, end_chars + 1):
            # Check if the last i characters of buffer match the first i characters of the tag
            if self.FULL_END_TAG.startswith(self._buffer[-i:]):
                self._partial_tag_buffer = self._buffer[-i:]
                break

        return filtered_content

    async def stop(self, frame: EndFrame):
        """Stop the NVIDIA LLM service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        if self._current_task and not self._current_task.done():
            await self.cancel_task(self._current_task)
            self._current_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA LLM service and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        if self._current_task and not self._current_task.done():
            await self.cancel_task(self._current_task)
            self._current_task = None

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Accumulate token usage metrics during processing.

        This method intercepts the incremental token updates from NVIDIA's API
        and accumulates them instead of passing each update to the metrics system.
        The final accumulated totals are reported at the end of processing.

        Args:
            tokens (LLMTokenUsage): The token usage metrics for the current chunk
                of processing, containing prompt_tokens and completion_tokens counts.
        """
        # Only accumulate metrics during active processing
        if not self._is_processing:
            return

        # Record prompt tokens the first time we see them
        if not self._has_reported_prompt_tokens and tokens.prompt_tokens > 0:
            self._prompt_tokens = tokens.prompt_tokens
            self._has_reported_prompt_tokens = True

        # Update completion tokens count if it has increased
        if tokens.completion_tokens > self._completion_tokens:
            self._completion_tokens = tokens.completion_tokens

    @traced_llm
    async def _process_context_and_frames(self, context: LLMContext):
        """Process context and handle start/end frames with metrics."""
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            # Start first sentence timing
            self._first_sentence_detected = False
            self._first_sentence_start_time = time.time()
            await self.start_processing_metrics()
            await self._process_context(context)
        except httpx.TimeoutException:
            await self._call_event_handler("on_completion_timeout")
        finally:
            await self.stop_processing_metrics()

            # Fallback: if first sentence was never detected during streaming (single-sentence response),
            # the first sentence time equals the full processing time
            if (
                not self._filter_think_tokens
                and not self._first_sentence_detected
                and self._first_sentence_start_time is not None
            ):
                processing_time = time.time() - self._first_sentence_start_time
                logger.debug(f"{self} LLM first sentence generation time: {processing_time:.3f}")

            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame in the specified direction."""
        context = None
        if LLMContextFrame is not None and isinstance(frame, LLMContextFrame):
            context: LLMContext = frame.context
        elif isinstance(frame, LLMMessagesUpdateFrame):
            context = LLMContext(messages=frame.messages)
        elif isinstance(frame, UserImageRawFrame):
            image_msg = await LLMContext.create_image_message(
                role="user",
                format=frame.format,
                size=frame.size,
                image=frame.image,
                text=getattr(frame, "text", None),
            )
            context = LLMContext(messages=[image_msg])
        elif isinstance(frame, InterruptionFrame):
            await self._start_interruption()
            await self.stop_all_metrics()
            await self.push_frame(frame, direction)
            if self._current_task is not None and not self._current_task.done():
                await self.cancel_task(self._current_task)
                self._current_task = None
        else:
            # All other frames are processed by the base llm service
            await super().process_frame(frame, direction)

        if context:
            if self._current_task is not None and not self._current_task.done():
                await self.cancel_task(self._current_task)
                self._current_task = None
                logger.trace("Old Nvidia LLM task terminated")
            self._current_task = self.create_task(self._process_context_and_frames(context))
            logger.trace("New Nvidia LLM task created")

            self._current_task.add_done_callback(lambda _: setattr(self, "_current_task", None))
