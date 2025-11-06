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
Python version of the agent-to-agent demo script.
Cross-platform alternative to run_agent_demo.sh.

This script:
1. Starts two voice agent servers on different ports
2. Waits for them to initialize
3. Starts the WebSocket bridge to connect them
4. Handles graceful shutdown of all processes

Usage:
    python run_agent_demo.py
    
With options:
    python run_agent_demo.py --agent1-port 8765 --agent2-port 8766 --startup-delay 10
"""

import argparse
import asyncio
import signal
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger


class Colors:
    """ANSI color codes for terminal output."""

    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def print_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")


def print_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")


def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


class AgentDemo:
    """Manages the agent-to-agent communication demo."""

    def __init__(
        self,
        agent1_port: int = 8765,
        agent2_port: int = 8766,
        agent1_name: str = "Alice",
        agent2_name: str = "Bob",
        startup_delay: int = 5,
    ):
        self.agent1_port = agent1_port
        self.agent2_port = agent2_port
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.startup_delay = startup_delay

        self.agent1_process = None
        self.agent2_process = None
        self.bridge_process = None

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print_info(f"Received signal {signum}, initiating shutdown...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean up all running processes."""
        print_info("Cleaning up processes...")

        if self.agent1_process:
            print_info(f"Stopping {self.agent1_name} (PID: {self.agent1_process.pid})")
            self.agent1_process.terminate()
            try:
                self.agent1_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.agent1_process.kill()

        if self.agent2_process:
            print_info(f"Stopping {self.agent2_name} (PID: {self.agent2_process.pid})")
            self.agent2_process.terminate()
            try:
                self.agent2_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.agent2_process.kill()

        if self.bridge_process:
            print_info(f"Stopping Bridge (PID: {self.bridge_process.pid})")
            self.bridge_process.terminate()
            try:
                self.bridge_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.bridge_process.kill()

        print_success("Cleanup complete")

    def check_files(self) -> bool:
        """Check if required files exist."""
        required_files = [
            "bot_websocket_server.py",
            "bot_websocket_server_alt.py",
            "connect_two_agents.py",
        ]

        for filename in required_files:
            if not Path(filename).exists():
                print_error(f"{filename} not found in current directory")
                return False

        return True

    def start_agent1(self):
        """Start the first agent server."""
        print_info(f"Starting {self.agent1_name} on port {self.agent1_port}...")

        log_file = open("agent1_output.log", "w")
        self.agent1_process = subprocess.Popen(
            [sys.executable, "bot_websocket_server.py"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        print_success(f"{self.agent1_name} started (PID: {self.agent1_process.pid})")

    def start_agent2(self):
        """Start the second agent server."""
        print_info(f"Starting {self.agent2_name} on port {self.agent2_port}...")

        log_file = open("agent2_output.log", "w")
        self.agent2_process = subprocess.Popen(
            [sys.executable, "bot_websocket_server_alt.py", "--port", str(self.agent2_port)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        print_success(f"{self.agent2_name} started (PID: {self.agent2_process.pid})")

    def wait_for_initialization(self):
        """Wait for agents to initialize."""
        print_info(f"Waiting {self.startup_delay}s for agents to initialize...")

        for i in range(self.startup_delay):
            print(".", end="", flush=True)
            time.sleep(1)

        print()
        print_success("Agents initialized")

    def check_agents_running(self) -> bool:
        """Check if both agents are still running."""
        if self.agent1_process.poll() is not None:
            print_error(f"{self.agent1_name} failed to start. Check agent1_output.log for details.")
            return False

        if self.agent2_process.poll() is not None:
            print_error(f"{self.agent2_name} failed to start. Check agent2_output.log for details.")
            return False

        return True

    def start_bridge(self):
        """Start the WebSocket bridge."""
        print_info("Starting WebSocket bridge...")

        self.bridge_process = subprocess.Popen(
            [
                sys.executable,
                "connect_two_agents.py",
                "--agent1-url",
                f"ws://localhost:{self.agent1_port}",
                "--agent2-url",
                f"ws://localhost:{self.agent2_port}",
                "--agent1-name",
                self.agent1_name,
                "--agent2-name",
                self.agent2_name,
                "--filter-audio",
            ],
        )

        print_success(f"Bridge started (PID: {self.bridge_process.pid})")

    def run(self):
        """Run the complete demo."""
        # Print header
        print()
        print("=" * 80)
        print("  Voice Agent-to-Agent Communication Demo")
        print("=" * 80)
        print(f"  Agent 1: {self.agent1_name} (Port {self.agent1_port})")
        print(f"  Agent 2: {self.agent2_name} (Port {self.agent2_port})")
        print(f"  Startup Delay: {self.startup_delay}s")
        print("=" * 80)
        print()

        # Check required files
        if not self.check_files():
            return 1

        try:
            # Start Agent 1
            self.start_agent1()
            time.sleep(2)  # Brief pause for initialization

            # Start Agent 2
            self.start_agent2()

            # Wait for initialization
            self.wait_for_initialization()

            # Check if agents are running
            if not self.check_agents_running():
                return 1

            # Start bridge
            self.start_bridge()

            # Print status
            print()
            print_success("Demo is now running!")
            print()
            print("The two agents are now connected and can communicate with each other.")
            print()
            print("Logs:")
            print("  - Agent 1 output: agent1_output.log")
            print("  - Agent 2 output: agent2_output.log")
            print("  - Bridge output: agent_bridge.log")
            print()
            print("Press Ctrl+C to stop all processes")
            print()

            # Wait for bridge to finish (or until interrupted)
            self.bridge_process.wait()

        except KeyboardInterrupt:
            print_info("Interrupted by user")
        except Exception as e:
            print_error(f"Error running demo: {e}")
            return 1
        finally:
            self.cleanup()

        return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent-to-agent communication demo (Python version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--agent1-port",
        type=int,
        default=8765,
        help="Port for first agent (default: 8765)",
    )

    parser.add_argument(
        "--agent2-port",
        type=int,
        default=8766,
        help="Port for second agent (default: 8766)",
    )

    parser.add_argument(
        "--agent1-name",
        type=str,
        default="Alice",
        help="Name for first agent (default: Alice)",
    )

    parser.add_argument(
        "--agent2-name",
        type=str,
        default="Bob",
        help="Name for second agent (default: Bob)",
    )

    parser.add_argument(
        "--startup-delay",
        type=int,
        default=5,
        help="Delay in seconds before starting bridge (default: 5)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    demo = AgentDemo(
        agent1_port=args.agent1_port,
        agent2_port=args.agent2_port,
        agent1_name=args.agent1_name,
        agent2_name=args.agent2_name,
        startup_delay=args.startup_delay,
    )

    return demo.run()


if __name__ == "__main__":
    sys.exit(main())
