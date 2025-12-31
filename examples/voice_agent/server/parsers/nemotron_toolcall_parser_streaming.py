import json
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits
from typing import Optional, Union

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow
from pydantic import Field
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer

logger = init_logger(__name__)

ALPHANUMERIC = ascii_letters + digits


class NemotronToolCall(ToolCall):
    id: str = Field(default_factory=lambda: NemotronToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) == 9


def _is_fn_name_regex_support(model_tokenizer: AnyTokenizer) -> bool:
    return isinstance(model_tokenizer, MistralTokenizer) and model_tokenizer.version >= 11


@ToolParserManager.register_module("nemotron_json")
class NemotronToolParser(ToolParser):
    """
    Tool call parser for Nemotron-Nano-V2

    Used when --enable-auto-tool-choice --tool-call-parser nemotron_json are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []  # map what has been streamed for each tool so far to a list
        self.tool_args_emitted: list[bool] = []
        self.bot_token = "<TOOLCALL>"
        self.bot_token_id = self.vocab.get(self.bot_token)
        logger.info(f"Nemotron Tool Parser: bot_token: {self.bot_token}, bot_token_id: {self.bot_token_id}")
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        if _is_fn_name_regex_support(self.model_tokenizer):
            self.fn_name_regex = re.compile(r'([a-zA-Z0-9_-]+)(\{[\s\S]*?\})(?=\s*$|,|\s)', re.DOTALL)
        else:
            self.fn_name_regex = None

        # Buffer for partial tag sequences to disambiguate between normal content and
        # a forthcoming <TOOLCALL> or </TOOLCALL> tag in streaming.
        self._pending_tag_buffer: str = ""

    @staticmethod
    def _strip_trailing_auto_closers(chunk: str) -> str:
        """
        Remove parser auto-completed closing braces/brackets plus trailing whitespace.
        These should be flushed only when a tool call completes to avoid duplicate
        argument fragments.
        """
        idx = len(chunk)
        while idx > 0 and chunk[idx - 1] in " \t\r\n}]":
            idx -= 1
        # Remove trailing non-escaped double quotes (partial JSON auto-closes strings)
        while idx > 0 and chunk[idx - 1] == '"':
            # keep escaped quotes (\"), only strip bare ones
            if idx - 2 >= 0 and chunk[idx - 2] == '\\':
                break
            idx -= 1
        return chunk[:idx]

    @staticmethod
    def _common_prefix_len(left: str, right: str) -> int:
        """
        Return the length of the shared prefix between left and right strings.
        """
        max_len = min(len(left), len(right))
        idx = 0
        while idx < max_len and left[idx] == right[idx]:
            idx += 1
        return idx

    def _compute_arguments_delta(self, cur_arguments_json: str, end_of_call: bool) -> str:
        """
        Determine the incremental suffix to stream for the current tool call.
        Ensures we only emit monotonic chunks by trimming our tracked prefix to
        the longest common prefix with the latest JSON snapshot.
        """
        tool_idx = self.current_tool_id
        if tool_idx < 0 or tool_idx >= len(self.streamed_args_for_tool):
            return ""

        streamed_prefix = self.streamed_args_for_tool[tool_idx]
        had_any = self.tool_args_emitted[tool_idx] if tool_idx < len(self.tool_args_emitted) else False

        lcp_len = self._common_prefix_len(cur_arguments_json, streamed_prefix)
        if lcp_len != len(streamed_prefix):
            streamed_prefix = streamed_prefix[:lcp_len]
            self.streamed_args_for_tool[tool_idx] = streamed_prefix

        if (
            not had_any
            and not end_of_call
            and lcp_len == 0
            and cur_arguments_json.endswith('": ""}')
            and '": ""' in cur_arguments_json
        ):
            closing_pos = cur_arguments_json.rfind('": ""}')
            if closing_pos != -1:
                arguments_delta = cur_arguments_json[: closing_pos + 4]
            else:
                arguments_delta = cur_arguments_json
        else:
            arguments_delta = cur_arguments_json[lcp_len:]

        if not arguments_delta:
            return ""

        if not end_of_call:
            arguments_delta = self._strip_trailing_auto_closers(arguments_delta)

        if not had_any and not end_of_call and arguments_delta and arguments_delta.endswith('}'):
            arguments_delta = arguments_delta[:-1]
            if arguments_delta.endswith('"'):
                arguments_delta = arguments_delta[:-1]

        return arguments_delta

    def _visible_delta_outside_tool(
        self, delta_text: str, start_token: Optional[str], end_token: Optional[str]
    ) -> str:
        """
        Consume characters that could begin a tool tag. Only suppress the exact
        <TOOLCALL> / </TOOLCALL> sequences, and let everything else (e.g. </think>)
        pass through untouched.
        """
        if not delta_text:
            return delta_text

        visible: list[str] = []
        for ch in delta_text:
            if self._pending_tag_buffer or ch == '<':
                self._pending_tag_buffer += ch

                if start_token and start_token.startswith(self._pending_tag_buffer):
                    if self._pending_tag_buffer == start_token:
                        self._pending_tag_buffer = ""
                    continue

                if end_token and end_token.startswith(self._pending_tag_buffer):
                    if self._pending_tag_buffer == end_token:
                        self._pending_tag_buffer = ""
                    continue

                # Not a tool tag; flush buffered characters as normal content.
                visible.append(self._pending_tag_buffer)
                self._pending_tag_buffer = ""
            else:
                visible.append(ch)

        return "".join(visible)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if not isinstance(self.model_tokenizer, MistralTokenizer) and request.tools and request.tool_choice != 'none':
            # Do not skip special tokens when using chat template
            # with Mistral parser as TOOL_CALL token is needed
            # for tool detection.
            # Note: we don't want skip_special_tokens=False
            # with MistralTokenizer as it is incompatible
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response. Requires
        find-and-replacing single quotes with double quotes for JSON parsing,
        make sure your tool call arguments don't ever include quotes!
        """

        # case -- if a tool call token is not present, return a text response
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        # first remove the BOT token
        tool_content = model_output.replace(self.bot_token, "").strip()

        try:
            # we first try to directly load the json as parsing very nested
            # jsons is difficult
            try:
                if self.fn_name_regex:
                    matches = self.fn_name_regex.findall(tool_content)

                    function_call_arr = []
                    for match in matches:
                        fn_name = match[0]
                        args = match[1]

                        # fn_name is encoded outside serialized json dump
                        # only arguments are serialized
                        function_call_arr.append({"name": fn_name, "arguments": json.loads(args)})
                else:
                    function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # use a regex to find the part corresponding to the tool call.
                # NOTE: This use case should not happen if the model is trained
                # correctly. It's a easy possible fix so it's included, but
                # can be brittle for very complex / highly nested tool calls
                raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
                function_call_arr = json.loads(raw_tool_call)

            # Tool Call
            tool_calls: list[NemotronToolCall] = [
                NemotronToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(raw_function_call["arguments"], ensure_ascii=False),
                    ),
                )
                for raw_function_call in function_call_arr
            ]

            # get any content before  the tool call
            content = model_output.split(self.bot_token)[0]
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content if len(content) > 0 else None
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=tool_content)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        # if candidates tool call tokens are in the tokens generated so far, that
        # means we're parsing as tool calls now. Suppress streaming if we are
        # currently generating any prefix of the start or end tag.
        visible_delta_text = delta_text
        try:
            start_token = self.bot_token
            end_token = f"</{self.bot_token[1:]}" if self.bot_token.startswith('<') else None

            visible_delta_text = self._visible_delta_outside_tool(delta_text, start_token, end_token)
        except Exception:
            # Fallback to conservative checks in case of any issues
            if (
                current_text.endswith('<')
                or current_text.endswith('<T')
                or current_text.endswith('<TO')
                or current_text.endswith('<TOOL')
                or current_text.endswith('<TOOLCALL')
            ):
                return None

        # if the tool call token is not in the tokens generated so far, append
        # output to contents since it's not a tool
        if self.bot_token not in current_text:
            if visible_delta_text:
                return DeltaMessage(content=visible_delta_text)
            # still waiting on a potential tag, so emit nothing yet
            return None

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        end_of_call: bool = False
        try:

            # replace BOT token with empty string, and convert single quotes
            # to double to allow parsing as JSON since mistral uses single
            # quotes instead of double for tool calls
            parsable_arr = current_text.split(self.bot_token)[-1]

            # Check if we're at the end of the tool call
            if '</TOOLCALL>' in parsable_arr:
                end_of_call = True
                parsable_arr = parsable_arr.split('</TOOLCALL>')[0]

            # tool calls are generated in an array, so do partial JSON
            # parsing on the entire array
            try:
                tool_call_arr: list[dict] = partial_json_parser.loads(parsable_arr, flags)
            except (partial_json_parser.core.exceptions.MalformedJSON, json.JSONDecodeError, ValueError):
                return None

            current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return None

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:

                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    diff: Union[str, None] = current_tool_call.get("arguments")

                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(
                            self.streamed_args_for_tool[self.current_tool_id], ""
                        )
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                    else:
                        delta = None
                else:
                    delta = None
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self.tool_args_emitted.append(False)
                return delta

            # case: update an existing tool - this is handled below

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=NemotronToolCall.generate_random_id(),
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = None

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:

                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                cur_arguments = current_tool_call.get("arguments")

                if not cur_arguments and not prev_arguments:

                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error("INVARIANT - impossible to have arguments reset " "mid-arguments")
                    delta = None
                elif cur_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                    arguments_delta = self._compute_arguments_delta(cur_arguments_json, end_of_call)
                    if arguments_delta:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=arguments_delta).model_dump(
                                        exclude_none=True
                                    ),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                        self.tool_args_emitted[self.current_tool_id] = True
                    else:
                        # Do not flush final JSON here; let the serving layer
                        # compute a minimal remaining suffix on finish.
                        delta = None
                else:
                    # End-of-call or equal state; do not force a final flush here.
                    delta = None

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            self.prev_tool_call_arr = tool_call_arr
            # If we've reached the end of a tool call, flush any remaining
            # suffix (including a final '}') that hasn't been streamed yet.
            if end_of_call and self.current_tool_id >= 0:
                try:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments is not None:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        remaining_suffix = self._compute_arguments_delta(cur_args_json, end_of_call=True)

                        # Only send remaining suffix if it's non-empty and contains meaningful content
                        # (not just whitespace or single characters like closing braces)
                        if remaining_suffix and remaining_suffix.strip():
                            extra = DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=remaining_suffix).model_dump(exclude_none=True),
                            )
                            if delta is None:
                                delta = DeltaMessage(tool_calls=[extra])
                            else:
                                if getattr(delta, "tool_calls", None):
                                    delta.tool_calls.append(extra)
                                else:
                                    delta.tool_calls = [extra]
                            self.streamed_args_for_tool[self.current_tool_id] += remaining_suffix
                            self.tool_args_emitted[self.current_tool_id] = True
                except Exception as e:
                    # Failure to flush the remaining arguments suffix is non-fatal; log for debugging.
                    logger.warning(f"Error in flushing remaining suffix for tool call: {e}")

            return delta

        except Exception as e:
            logger.exception(f"Error trying to handle streaming tool call: {e}")
            return None
