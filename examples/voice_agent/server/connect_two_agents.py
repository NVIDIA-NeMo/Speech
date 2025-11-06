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
Script to connect two voice agents directly so they can communicate with each other.

This script uses the WebsocketBridge class to establish a connection between two
voice agent servers and forward messages between them bidirectionally.

Usage:
    python connect_two_agents.py --agent1-url ws://localhost:8765 --agent2-url ws://localhost:8766
    
Or with custom names:
    python connect_two_agents.py \
        --agent1-url ws://localhost:8765 --agent1-name "Alice" \
        --agent2-url ws://localhost:8766 --agent2-name "Bob" \
        --log-messages --filter-audio
"""

import argparse
import asyncio
import json
import signal
import sys

from loguru import logger
from websocket_bridge import WebsocketBridge


def setup_logging():
    """Configure logging output."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
    )
    logger.add("agent_bridge.log", rotation="1 day", level="DEBUG")


async def send_client_ready(ws, agent_name: str):
    """
    Send a client ready message to initialize the agent.
    This mimics what a Pipecat client would send.
    """
    try:
        client_ready_message = {"type": "rtvi-client-ready", "version": "0.1", "config": []}
        await ws.send(json.dumps(client_ready_message))
        logger.info(f"Sent client ready message to {agent_name}")
    except Exception as e:
        logger.error(f"Error sending client ready message to {agent_name}: {e}")


async def initialize_agents(bridge: WebsocketBridge, delay: float = 1.0):
    """
    Initialize both agents by sending client ready messages.

    Args:
        bridge: The WebsocketBridge instance
        delay: Delay in seconds between sending messages to each agent
    """
    if not bridge.agent1_ws or not bridge.agent2_ws:
        raise RuntimeError("Bridge must be connected before initialization")

    logger.info("Initializing agents...")

    # Send client ready to agent 1
    await send_client_ready(bridge.agent1_ws, bridge.agent1_name)

    # Small delay to ensure proper initialization
    await asyncio.sleep(delay)

    # Send client ready to agent 2
    await send_client_ready(bridge.agent2_ws, bridge.agent2_name)

    logger.info("Agents initialized and ready to communicate")


async def run_bridge_with_initialization(
    agent1_url: str,
    agent2_url: str,
    agent1_name: str = "Agent1",
    agent2_name: str = "Agent2",
    log_messages: bool = True,
    filter_audio: bool = False,
    init_delay: float = 1.0,
):
    """
    Create and run a bridge between two agents with proper initialization.

    Args:
        agent1_url: WebSocket URL for the first agent
        agent2_url: WebSocket URL for the second agent
        agent1_name: Name for the first agent (for logging)
        agent2_name: Name for the second agent (for logging)
        log_messages: Whether to log message traffic
        filter_audio: Whether to filter audio frames from logging
        init_delay: Delay between agent initializations
    """
    bridge = WebsocketBridge(
        agent1_url=agent1_url,
        agent2_url=agent2_url,
        agent1_name=agent1_name,
        agent2_name=agent2_name,
        log_messages=log_messages,
        filter_audio=filter_audio,
    )

    try:
        # Connect to both agents
        await bridge.connect()

        # Initialize agents
        await initialize_agents(bridge, delay=init_delay)

        # Start forwarding messages
        logger.info(f"Bridge established between {agent1_name} and {agent2_name}")
        logger.info("The agents will now communicate with each other...")
        logger.info("Press Ctrl+C to stop")

        await bridge.run()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping bridge...")
    except Exception as e:
        logger.error(f"Error running bridge: {e}")
    finally:
        await bridge.disconnect()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Connect two voice agents to communicate with each other",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect two agents on different ports
  python connect_two_agents.py --agent1-url ws://localhost:8765 --agent2-url ws://localhost:8766

  # With custom names and logging options
  python connect_two_agents.py \\
      --agent1-url ws://localhost:8765 --agent1-name "Alice" \\
      --agent2-url ws://localhost:8766 --agent2-name "Bob" \\
      --log-messages --filter-audio

  # Minimal logging
  python connect_two_agents.py \\
      --agent1-url ws://localhost:8765 \\
      --agent2-url ws://localhost:8766 \\
      --no-log-messages
        """,
    )

    parser.add_argument(
        "--agent1-url",
        type=str,
        default="ws://localhost:8765",
        help="WebSocket URL for the first agent (default: ws://localhost:8765)",
    )

    parser.add_argument(
        "--agent2-url",
        type=str,
        default="ws://localhost:8766",
        help="WebSocket URL for the second agent (default: ws://localhost:8766)",
    )

    parser.add_argument(
        "--agent1-name",
        type=str,
        default="Agent1",
        help="Name for the first agent (for logging)",
    )

    parser.add_argument(
        "--agent2-name",
        type=str,
        default="Agent2",
        help="Name for the second agent (for logging)",
    )

    parser.add_argument(
        "--log-messages",
        action="store_true",
        default=True,
        help="Log message traffic (default: True)",
    )

    parser.add_argument(
        "--no-log-messages",
        action="store_false",
        dest="log_messages",
        help="Disable message traffic logging",
    )

    parser.add_argument(
        "--filter-audio",
        action="store_true",
        default=False,
        help="Filter audio frames from logging (default: False)",
    )

    parser.add_argument(
        "--init-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between agent initializations (default: 1.0)",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    setup_logging()
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Voice Agent Bridge - Connecting Two Agents")
    logger.info("=" * 80)
    logger.info(f"{args.agent1_name}: {args.agent1_url}")
    logger.info(f"{args.agent2_name}: {args.agent2_url}")
    logger.info(f"Log messages: {args.log_messages}")
    logger.info(f"Filter audio: {args.filter_audio}")
    logger.info("=" * 80)

    await run_bridge_with_initialization(
        agent1_url=args.agent1_url,
        agent2_url=args.agent2_url,
        agent1_name=args.agent1_name,
        agent2_name=args.agent2_name,
        log_messages=args.log_messages,
        filter_audio=args.filter_audio,
        init_delay=args.init_delay,
    )


if __name__ == "__main__":
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(main())
