# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

__all__ = ['TokenizerSpec']


class TokenizerSpec(ABC):
    """
    Inherit this class to implement a new tokenizer.
    """

    @abstractmethod
    def text_to_tokens(self, text):
        pass

    @abstractmethod
    def tokens_to_text(self, tokens):
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens):
        pass

    @abstractmethod
    def ids_to_tokens(self, ids):
        pass

    @abstractmethod
    def text_to_ids(self, text):
        pass

    @abstractmethod
    def ids_to_text(self, ids):
        pass

    def add_special_tokens(self, special_tokens: List[str]):
        raise NotImplementedError("To be implemented")

    @property
    def name(self):
        return type(self).__name__

    @property
    def unique_identifiers(self):
        """Property required for use with megatron-core datasets."""
        return OrderedDict({"class": f"{type(self).__module__}.{type(self).__qualname__}"})

    @property
    def cls(self):
        """Property alias to match MegatronTokenizer; returns cls_id if available."""
        if hasattr(self, 'cls_id'):
            return self.cls_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'cls' or 'cls_id'")

    @property
    def sep(self):
        """Property alias to match MegatronTokenizer; returns sep_id if available."""
        if hasattr(self, 'sep_id'):
            return self.sep_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'sep' or 'sep_id'")

    @property
    def pad(self):
        """Property alias to match MegatronTokenizer; returns pad_id if available."""
        if hasattr(self, 'pad_id'):
            return self.pad_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'pad' or 'pad_id'")

    @property
    def eod(self):
        """Property alias to match MegatronTokenizer; returns eod_id if available."""
        if hasattr(self, 'eod_id'):
            return self.eod_id
        if hasattr(self, 'eos_id'):
            # Default to end-of-sentence id if end-of-document is not defined.
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eod', 'eod_id', 'eos', or 'eos_id'")

    @property
    def bos(self):
        """Property alias to match MegatronTokenizer; returns bos_id if available."""
        if hasattr(self, 'bos_id'):
            return self.bos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'bos' or 'bos_id'")

    @property
    def eos(self):
        """Property alias to match MegatronTokenizer; returns eos_id if available."""
        if hasattr(self, 'eos_id'):
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eos' or 'eos_id'")

    @property
    def mask(self):
        """Property alias to match MegatronTokenizer; returns mask_id if available."""
        if hasattr(self, 'mask_id'):
            return self.mask_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'mask' or 'mask_id'")
