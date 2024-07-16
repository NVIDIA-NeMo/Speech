from abc import ABC
from enum import Enum
from functools import lru_cache
from typing import Any, Type

import torch

from nemo.collections.common.tokenizers import AggregateTokenizer, TokenizerSpec

PREAMBLE_ROLE = "preamble"

# Slots used to define when special tokens bos/eos should be inserted.
# These are special in the sense of how sentencepiece defines special tokens:
# They have to be specially inserted into the token sequence, and if they appear in the tokenized string,
# SPE wouldn't use the special token ids but rather tokenize them as if they were normal strings.
# We mimic SPE's behavior if these special slots are present in the template definition.
# To achieve that, insert |bos| / |eos| at the beginning/end of template.
# E.g., inserting only bos in llama2 user role: "template": "|bos|[INST] |message| [\INST]"
BOS_SLOT = "|bos|"
EOS_SLOT = "|eos|"


class BaseModalityType:
    @staticmethod
    def matches(value: Any) -> bool:
        raise NotImplementedError

    def __repr__(self):
        return f"Modality.{self.__class__.__name__}()"


class Text(BaseModalityType):
    """Modality for text values."""

    @staticmethod
    def matches(value: str) -> bool:
        return isinstance(value, str)


class TextLiteral(BaseModalityType):
    def __init__(self, *items):
        self.allowed_values = items

    def matches(self, value: str) -> bool:
        return isinstance(value, str) and value in self.allowed_values

    def __repr__(self):
        return f"Modality.{self.__class__.__name__}(allowed_values={self.allowed_values})"


class Modality:
    """
    Modalities supported as PromptFormatter slot values.
    """

    Text = Text
    TextLiteral = TextLiteral


class PromptFormatter(ABC):
    """
    :class:`~nemo.collections.common.prompts.formatter.PromptFormatter` is intended to simplify
    working with various prompt format templates and encoding them into token ID tensors.

    It assumes a dialog-like structure, which is a list of turns, with each turn assigned to a role.
    Sub-classes of PromptFormatter define turn templates for each role under TEMPLATE class attribute.
    Each template may define some constant parts (e.g. begin-of-turn or end-of-turn tokens, whitespaces, etc.)
    and variable parts which we call "slots", that will be provided by the user during training or inference.

    A role is typically "user" and "assistant", and some popular models also use a "system" role.
    Other roles may be defined as well. We expect the role corresponding to the model's responses
    will be registered under class attribute called OUTPUT_ROLE.
    We reserve a special "preamble" role with no slots that will be inserted at the beginning of
    the formatted prompt, if "preamble" is present in TEMPLATE.

    A turn is a dict with keys "role" and "slots", where "slots" are a dict that maps slot names
    to values that should be filled in the template.
    For example, a user role template may be ``"Question: |message|"`` and corresponding ``slots`` would then be
    ``{"message": "What time is it?"}``.

    There is a special slot called ``|prompt_language|`` that's used to select the sub-tokenizer in
    :class:`~nemo.collections.common.tokenizers.aggregate_tokenizer.AggregateTokenizer`.
    It's only used when the tokenizer is aggregate; otherwise it's discarded.

    PromptFormatter supports constructing prompts for training (complete context and answers)
    and for inference (context-only).
    Training/inference is determined automatically; if the last role in a dialog is the OUTPUT_ROLE,
    that's an 'asked-and-answered' scenario, so we assume it's inteded for training.
    We'll create a dict with tokenized results available under the following keys:

    * ``context_ids`` (all turns minus last one),
    * ``answer_ids`` (last turn)
    * ``input_ids`` (previous two values concatenated)
    * ``mask`` (boolean mask tensor of the same lenth as ``input_ids`` that's set to True on OUTPUT_ROLE turns)

    Typically, the user will use the ``encode_dialog`` method providing a list of turns to it.
    Example showing how to construct model inputs/outputs for training::

        >>> formatter = PromptFormatter(tokenizer)
        ... encoded_for_training = formatter.encode_dialog(
        ...     turns=[
        ...         {"role": "user", "slots": {"message": "What time is it?"}},
        ...         {"role": "assistant", "slots": {"message": "Ten o'clock."}},
        ...         {"role": "user", "slots": {"message": "PM or AM?"}},
        ...         {"role": "assistant", "slots": {"message": "AM, naturally! It's bright outside"}},
        ...     ]
        ... )

    Another example that shows how to use the same method to generate prompts for inference::


        >>> formatter = PromptFormatter(tokenizer)
        ... encoded_for_training = formatter.encode_dialog(
        ...     turns=[
        ...         {"role": "user", "slots": {"message": "What time is it?"}},
        ...         {"role": "assistant", "slots": {"message": "Ten o'clock."}},
        ...         {"role": "user", "slots": {"message": "PM or AM?"}},
        ...     ]
        ... )

    """

    # Used to support AggregateTokenizer; this key selects the right sub-tokenizer for each turn.
    PROMPT_LANGUAGE_SLOT = "prompt_language"

    # Subclasses will be registered under this name, to be used via PromptFormatter.resolve(name).
    NAME = None

    # Template is a dict that maps:
    # * from a role name string (system/user/assistant/etc)
    # * to a dict with keys
    #   * "template" that has a string value (the prompt template)
    #   * "slots" that has a value of dict[str, Modality]
    #       * keys of slots are the names of formattable slots in the prompt template
    #       * values of slots are :class:`Modality` objects that can be used to check
    #           whether a specific value conforms to a given modality requirements
    #           (e.g., Modality.Text may expect string objects).
    # Template is intended to be defined by the child classes.
    TEMPLATE = None

    # Turns under this role indicate responses by the model; if the last turn in
    # PromptFormatter.encode_dialog() ends with this role, it indicates a training example.
    OUTPUT_ROLE = None

    # Internal reserved field.
    _REGISTERED_FORMATTERS = {}

    def __init__(self, tokenizer: TokenizerSpec, defaults: list[dict] | None = None) -> None:
        self.tokenizer = tokenizer
        self._defaults = defaults if defaults is not None else []
        self._validate_defaults()

    def __init_subclass__(cls, **kwargs) -> None:
        ERR = "PromptFormatter subclass definition error:"
        if cls.__name__ not in cls._REGISTERED_FORMATTERS:
            for attr in ("NAME", "TEMPLATE", "OUTPUT_ROLE"):
                assert (
                    getattr(cls, attr, None) is not None
                ), f"{ERR} PromptFormatter subclass {cls} did not define a class attribute {attr}"
            assert cls.NAME not in cls._REGISTERED_FORMATTERS, (
                f"Cannot register {cls.__name__} under {cls.NAME}: another prompt formatter of type "
                f"{cls._REGISTERED_FORMATTERS[cls.NAME]} has already been registered under this name."
            )
            cls._REGISTERED_FORMATTERS[cls.NAME] = cls
        if "preamble" in cls.TEMPLATE:
            assert (
                len(cls.TEMPLATE["preamble"].get("slots", [])) == 0
            ), f"{ERR} Slots are not allowed for preamble template, but we found: '{cls.TEMPLATE['preamble']}'"
        for role in cls.get_roles():
            template = cls.get_template(role)
            for slot in cls.get_slots(role):
                assert (
                    _mangled(slot) in template
                ), f"{ERR} Slot '{slot}' not found in template '{template}' for role '{role}'"
        super().__init_subclass__(**kwargs)

    @classmethod
    def resolve(cls, name: str) -> Type["PromptFormatter"]:
        if name not in cls._REGISTERED_FORMATTERS:
            raise RuntimeError(
                f"Unknown prompt formatter: '{name}' (known formats: {', '.join(cls._REGISTERED_FORMATTERS.keys())})"
            )
        return cls._REGISTERED_FORMATTERS[name]

    @classmethod
    @lru_cache(1)
    def get_roles(cls) -> list[str]:
        return list(cls.TEMPLATE.keys())

    @classmethod
    def get_slots(cls, role: str) -> dict[str, Modality]:
        # returns a copy to avoid accidential mutation of a global object by the user
        return cls.TEMPLATE[role].get("slots", {}).copy()

    @classmethod
    def get_template(cls, role: str) -> str:
        return cls.TEMPLATE[role]["template"]

    def get_default_dialog_slots(self) -> list[dict]:
        """
        Returns a list of dialog turns that can be used as a skeleton to fill with actual slot values.
        If ``PromptFormatter`` was initialized with ``defaults`` argument, this method will return the
        defaults. Otherwise, every slot is pre-filled with ``None``.
        """

        def _get_default_for_role(role: str) -> dict:
            for turn in self._defaults:
                if turn["role"] == role:
                    return turn
            return {}

        return [
            {
                "role": role,
                "slots": {
                    slot: _get_default_for_role(role).get("slots", {}).get(slot) for slot in self.get_slots(role)
                },
            }
            for role in self.get_roles()
            if role != self.OUTPUT_ROLE
        ]

    def encode_turn(
        self, prompt_template: str, expected_slots: dict[str, Modality], slot_values: dict[str, Any]
    ) -> list[int]:
        prompt = prompt_template
        for slot in expected_slots:
            # For the final substitution of 'slot' in the template we have to mangle it to '|slot|' anyway,
            # but 'slot' form enables to use valid python identifiers as **kwargs
            # for passing slots around in user functions.
            value = slot_values.get(slot)
            assert value is not None, f"Missing required {slot=} in {slot_values=} for {prompt_template=}"
            prompt = prompt.replace(_mangled(slot), value)
        return self._apply_tokenizer(prompt, lang=slot_values.get(self.PROMPT_LANGUAGE_SLOT))

    def encode_dialog(self, turns: list[dict]) -> dict[str, torch.Tensor]:
        assert len(turns) > 0, "Empty dialog is not supported."
        roles = self.get_roles()

        turn_tokens = []
        turn_token_counts = []
        turn_mask_values = []

        if "preamble" in self.TEMPLATE:
            preamble_turns = [idx for idx, t in enumerate(turns) if t["role"] == "preamble"]
            if not preamble_turns:
                turns = [{"role": "preamble", **self.TEMPLATE["preamble"]}] + turns
            else:
                assert (
                    len(preamble_turns) == 1 and preamble_turns[0] == 0
                ), f"Preamble can only be presented at turn 0, but we found preamble turns at indexes {preamble_turns}."

        for turn in turns:
            assert "role" in turn, f"A turn must have have a 'role' key. We received {turn=}"
            role = turn["role"]
            assert role in roles, f"Found turn with {role=}, but availables roles are {roles}"
            expected_slots = self.get_slots(role)
            slot_values = turn.get("slots", {})
            if expected_slots:
                assert (
                    slot_values
                ), f"A turn for role {role} must have have a non-empty value under 'slots' key. We received {turn=}"
                self._validate_slot_values(expected_slots, slot_values)
            template = self.get_template(role)
            tokens = self.encode_turn(template, expected_slots, slot_values)
            turn_tokens.extend(tokens)
            turn_token_counts.append(len(tokens))
            turn_mask_values.append(role == self.OUTPUT_ROLE)

        ans = {"input_ids": torch.tensor(turn_tokens, dtype=torch.long)}
        if turn_mask_values[-1]:
            # The last turn comes from OUTPUT_ROLE, i.e. it's a response from the system.
            # This indicates it's a training example for which we provide context/answer/mask.
            ans["context_ids"] = ans["input_ids"][: -turn_token_counts[-1]]
            ans["answer_ids"] = ans["input_ids"][-turn_token_counts[-1] :]
            ans["mask"] = torch.tensor(
                [
                    turn_mask_values[turn_idx]
                    for turn_idx, turn_len in enumerate(turn_token_counts)
                    for _ in range(turn_len)
                ],
                dtype=torch.bool,
            )
        else:
            ans["context_ids"] = ans["input_ids"]  # context == input for inference
        return ans

    def _apply_tokenizer(self, text: str, lang: str | None = None) -> list[int]:
        # Check if the tokenizer is aggregate and perform extra checks.
        is_agg = isinstance(self.tokenizer, AggregateTokenizer)
        if is_agg:
            assert lang is not None, (
                f"Missing key '{self.PROMPT_LANGUAGE_SLOT}' in slot_values -- cannot resolve "
                f"the correct sub-tokenizer in the aggregate tokenizer."
            )

        # Strip bos/eos if present and remember to apply them later.
        has_bos = text.startswith(BOS_SLOT)
        has_eos = text.endswith(EOS_SLOT)
        if has_bos:
            text = text[len(BOS_SLOT) :]
        if has_eos:
            text = text[: -len(EOS_SLOT)]

        # Tokenize, selecting the right API depending on aggregate/normal tokenizer.
        if is_agg:
            tokens = self.tokenizer.text_to_ids(text, lang)
        else:
            tokens = self.tokenizer.text_to_ids(text)

        # Lazily look up bos/eos and apply them. Lazy has the advantage that if a tokenizer
        # doesn't define bos/eos and the prompt format does not request them, everything just works.
        if has_eos:
            eos_id = self.tokenizer.get_eos(lang) if is_agg else self.tokenizer.eos
            tokens.append(eos_id)
        if has_bos:
            bos_id = self.tokenizer.get_bos(lang) if is_agg else self.tokenizer.bos
            tokens = [bos_id] + tokens

        return tokens

    def _validate_slot_values(self, expected: dict[str, Modality], received: dict[str, Any]) -> None:
        missing = set(expected) - set(received)
        assert not missing, f"The following slot values were not provided: {missing}"
        for slot in expected:
            expected_modality = expected[slot]
            value = received[slot]
            assert expected_modality.matches(
                value
            ), f"{slot=} received {value=} which does not match modality {expected_modality}"

    def _validate_defaults(self):
        if not self._defaults:
            return

        err = "Error in default prompt definition:"
        assert isinstance(self._defaults, list)
        for turn in self._defaults:
            assert isinstance(turn, dict)
            assert "role" in turn, f"{err} Missing required 'role' key. We received {turn=}"
            role = turn["role"]
            assert role in self.get_roles(), (
                f"{err} Invalid {role=} in {turn=} - " f"supported roles are: {self.get_roles()}."
            )
            if expected_slots := self.get_slots(role):
                assert "slots" in turn, (
                    f"{err} Missing required 'slots' key in {turn=} - "
                    f"we expected the following slots to be provided: {expected_slots}."
                )
                for slot in turn["slots"]:
                    assert slot in expected_slots, (
                        f"{err} Invalid {slot=} in {turn=}. "
                        f"The following slots are supported for {role=}: {expected_slots}"
                    )


def _mangled(slot: str) -> str:
    if not (slot[0] == "|" and slot[-1] == "|"):
        return f"|{slot}|"
    return slot


def _unmangled(slot: str) -> str:
    if slot[0] == "|" and slot[-1] == "|":
        return slot[1:-1]
    return slot
