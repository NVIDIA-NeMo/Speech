# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from tqdm import tqdm


class ProgressBar:
    """
    Base class for progress bars.
    """

    def __init__(self, value: float = 0.0, total: float = 1.0):
        """
        Initialize the ProgressBar.
        Args:
            value: (float) Initial value.
            total: (float) Total value. Must be greater than zero.
        """
        if total <= 0:
            raise ValueError("Total must be greater than zero.")
        if value < 0 or value > total:
            raise ValueError("Initial value must be between 0 and total.")

        self.value = value
        self.total = total
        self.start_value = value

    def restart(self) -> None:
        """Restart progress from the initial value."""
        self.value = self.start_value

    def increment(self, by: float) -> None:
        """
        Increase progress but do not exceed total.
        Args:
            by: (float) Amount to increment.
        """
        self.value = min(self.value + by, self.total)

    def update_bar(self, by: float) -> None:
        """
        Update progress and call update.
        Args:
            by: (float) Amount to increment.
        """
        self.increment(by)
        self.update()

    def finish(self) -> None:
        """Complete progress bar."""
        self.value = self.total
        self.update(True)

    def update(self, is_end: bool = False) -> None:
        """
        Abstract method for updating the progress bar.
        Args:
            is_end: (bool) Whether the progress bar is at the end.
        """
        raise NotImplementedError("Subclasses must implement update method.")


class TQDMProgressBar(ProgressBar):
    """TQDM progress bar wrapper."""

    def __init__(self, value: float = 0.0, total: float = 1.0):
        """
        Initialize the TQDMProgressBar.
        Args:
            value: (float) Initial value.
            total: (float) Total value.
        """
        super().__init__(value, total)
        self.bar = tqdm(total=self.total, bar_format='{l_bar}{bar}')
        self.prev_value = value

    def update(self, is_end: bool = False) -> None:
        """
        Update tqdm progress bar.
        Args:
            is_end: (bool) Whether the progress bar is at the end.
        """
        increment = self.value - self.prev_value
        if increment > 0:
            self.bar.update(increment)
        self.prev_value = self.value

        if is_end:
            self.bar.close()
