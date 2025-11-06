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

"""
Advanced WebsocketBridge Example

This example demonstrates:
1. Extending the WebsocketBridge class
2. Adding custom message filtering
3. Message transformation/inspection
4. Conversation analytics
5. Conditional message forwarding

Use this as a starting point for more sophisticated agent-to-agent setups.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

from loguru import logger
from websocket_bridge import WebsocketBridge


class AnalyticsWebsocketBridge(WebsocketBridge):
    """
    Extended WebsocketBridge with analytics and message inspection.

    Features:
    - Track message counts and types
    - Monitor conversation duration
    - Filter sensitive content
    - Log conversation summaries
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Analytics
        self.start_time = None
        self.message_count = {"agent1_to_agent2": 0, "agent2_to_agent1": 0}
        self.message_types = {}
        self.conversation_events = []

        # Optional: word filtering
        self.blocked_words = set()  # Add words to block if needed

    async def connect(self):
        """Connect and start analytics."""
        await super().connect()
        self.start_time = datetime.now()
        logger.info("Analytics tracking started")

    def _track_message(self, message, direction: str):
        """Track message for analytics."""
        self.message_count[direction] += 1

        # Try to parse message type
        try:
            if isinstance(message, str):
                data = json.loads(message)
                msg_type = data.get("type", "unknown")
                self.message_types[msg_type] = self.message_types.get(msg_type, 0) + 1
        except (json.JSONDecodeError, AttributeError):
            self.message_types["binary"] = self.message_types.get("binary", 0) + 1

    def _should_forward_message(self, message) -> bool:
        """
        Determine if a message should be forwarded.
        Override this to implement custom filtering logic.
        """
        # Example: Block messages containing sensitive words
        if isinstance(message, str) and self.blocked_words:
            try:
                data = json.loads(message)
                text = str(data).lower()
                for word in self.blocked_words:
                    if word.lower() in text:
                        logger.warning(f"Blocked message containing: {word}")
                        return False
            except:
                pass

        return True

    def _transform_message(self, message, source_name: str, dest_name: str):
        """
        Transform a message before forwarding.
        Override this to modify messages in transit.
        """
        # Example: Add metadata to JSON messages
        if isinstance(message, str):
            try:
                data = json.loads(message)
                # Add routing metadata
                if "metadata" not in data:
                    data["metadata"] = {}
                data["metadata"]["forwarded_by"] = "WebsocketBridge"
                data["metadata"]["source"] = source_name
                data["metadata"]["destination"] = dest_name
                data["metadata"]["timestamp"] = datetime.now().isoformat()
                return json.dumps(data)
            except (json.JSONDecodeError, TypeError):
                pass

        return message

    async def _forward_messages(self, source_ws, dest_ws, source_name: str, dest_name: str):
        """Enhanced message forwarding with analytics and filtering."""
        direction = f"{source_name.lower()}_to_{dest_name.lower()}"

        try:
            async for message in source_ws:
                # Track the message
                self._track_message(message, direction)

                # Check if should forward
                if not self._should_forward_message(message):
                    continue

                # Log message
                if self._should_log_message(message):
                    preview = self._format_message_preview(message)
                    logger.debug(f"{source_name} -> {dest_name}: {preview}")

                # Transform message (if needed)
                transformed = self._transform_message(message, source_name, dest_name)

                # Forward the message
                if isinstance(transformed, bytes):
                    await dest_ws.send(transformed)
                else:
                    await dest_ws.send(transformed)

        except Exception as e:
            logger.error(f"Error in forwarding: {e}")
            raise

    def get_analytics_summary(self) -> dict:
        """Get analytics summary."""
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        return {
            "duration_seconds": duration,
            "total_messages": sum(self.message_count.values()),
            "messages_by_direction": self.message_count,
            "messages_by_type": self.message_types,
            "average_messages_per_minute": (sum(self.message_count.values()) / duration * 60) if duration > 0 else 0,
        }

    async def disconnect(self):
        """Disconnect and print analytics."""
        await super().disconnect()

        # Print analytics summary
        summary = self.get_analytics_summary()
        logger.info("=" * 80)
        logger.info("CONVERSATION ANALYTICS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info(f"Total Messages: {summary['total_messages']}")
        logger.info(f"Average Rate: {summary['average_messages_per_minute']:.2f} messages/minute")
        logger.info("")
        logger.info("Messages by Direction:")
        for direction, count in summary['messages_by_direction'].items():
            logger.info(f"  {direction}: {count}")
        logger.info("")
        logger.info("Messages by Type:")
        for msg_type, count in summary['messages_by_type'].items():
            logger.info(f"  {msg_type}: {count}")
        logger.info("=" * 80)


class ConversationRecorderBridge(WebsocketBridge):
    """
    WebsocketBridge that records the conversation to a file.
    """

    def __init__(self, *args, output_file: str = "conversation_log.jsonl", **kwargs):
        super().__init__(*args, **kwargs)
        self.output_file = output_file
        self.log_handle = None

    async def connect(self):
        """Connect and open log file."""
        await super().connect()
        self.log_handle = open(self.output_file, "w")
        logger.info(f"Recording conversation to {self.output_file}")

    def _record_message(self, message, source: str, destination: str):
        """Record a message to the log file."""
        if not self.log_handle:
            return

        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "destination": destination,
            }

            if isinstance(message, str):
                try:
                    # Try to parse as JSON
                    record["message"] = json.loads(message)
                except json.JSONDecodeError:
                    record["message"] = message
            else:
                record["message"] = f"<binary data, {len(message)} bytes>"

            # Write as JSON line
            self.log_handle.write(json.dumps(record) + "\n")
            self.log_handle.flush()

        except Exception as e:
            logger.error(f"Error recording message: {e}")

    async def _forward_messages(self, source_ws, dest_ws, source_name: str, dest_name: str):
        """Forward messages with recording."""
        try:
            async for message in source_ws:
                # Record the message
                self._record_message(message, source_name, dest_name)

                # Log if needed
                if self._should_log_message(message):
                    preview = self._format_message_preview(message)
                    logger.debug(f"{source_name} -> {dest_name}: {preview}")

                # Forward the message
                if isinstance(message, bytes):
                    await dest_ws.send(message)
                else:
                    await dest_ws.send(message)

        except Exception as e:
            logger.error(f"Error in forwarding: {e}")
            raise

    async def disconnect(self):
        """Disconnect and close log file."""
        if self.log_handle:
            self.log_handle.close()
            logger.info(f"Conversation recording saved to {self.output_file}")
        await super().disconnect()


async def example_analytics_bridge():
    """Example: Using analytics bridge."""
    logger.info("Starting Analytics Bridge Example")

    bridge = AnalyticsWebsocketBridge(
        agent1_url="ws://localhost:8765",
        agent2_url="ws://localhost:8766",
        agent1_name="Agent1",
        agent2_name="Agent2",
        log_messages=True,
        filter_audio=True,
    )

    try:
        await bridge.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Analytics are printed in disconnect()
        pass


async def example_recorder_bridge():
    """Example: Using conversation recorder bridge."""
    logger.info("Starting Recorder Bridge Example")

    bridge = ConversationRecorderBridge(
        agent1_url="ws://localhost:8765",
        agent2_url="ws://localhost:8766",
        agent1_name="Agent1",
        agent2_name="Agent2",
        output_file="conversation_recording.jsonl",
        log_messages=True,
        filter_audio=True,
    )

    try:
        await bridge.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


async def example_combined():
    """Example: Combining multiple features."""
    logger.info("Starting Combined Features Example")

    class CombinedBridge(AnalyticsWebsocketBridge, ConversationRecorderBridge):
        """Bridge with both analytics and recording."""

        pass

    bridge = CombinedBridge(
        agent1_url="ws://localhost:8765",
        agent2_url="ws://localhost:8766",
        agent1_name="Alice",
        agent2_name="Bob",
        output_file="full_conversation_log.jsonl",
        log_messages=True,
        filter_audio=True,
    )

    # Optional: Add word filtering
    bridge.blocked_words = {"password", "secret", "confidential"}

    try:
        await bridge.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    import sys

    # Choose which example to run
    examples = {
        "analytics": example_analytics_bridge,
        "recorder": example_recorder_bridge,
        "combined": example_combined,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        example = examples[sys.argv[1]]
    else:
        print("Usage: python advanced_bridge_example.py [analytics|recorder|combined]")
        print()
        print("Examples:")
        print("  python advanced_bridge_example.py analytics  - Run with analytics")
        print("  python advanced_bridge_example.py recorder   - Run with conversation recording")
        print("  python advanced_bridge_example.py combined   - Run with all features")
        print()
        print("Defaulting to 'combined' example...")
        example = example_combined

    asyncio.run(example())
