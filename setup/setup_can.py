#!/usr/bin/env python3
"""
CAN interface setup script for RobStride.
Configures CAN0, CAN1, and CAN2 interfaces with 1Mbps bitrate.
"""

import subprocess
import sys

def run_command(cmd):
    """Execute a command with sudo."""
    try:
        full_cmd = f"sudo {cmd}"
        result = subprocess.run(full_cmd, shell=True, check=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd} (failed with code {e.returncode})")
        return False

def main():
    """Set up CAN interfaces."""
    print("Setting up CAN interfaces...")
    
    commands = [
        "ip link set can0 type can bitrate 1000000",
        "ip link set can1 type can bitrate 1000000",
        "ip link set can2 type can bitrate 1000000",
        "ip link set can0 up",
        "ip link set can1 up",
        "ip link set can2 up",
    ]
    
    failed = []
    for cmd in commands:
        if not run_command(cmd):
            failed.append(cmd)
    
    if failed:
        print(f"\n{len(failed)} command(s) failed:")
        for cmd in failed:
            print(f"  - {cmd}")
        sys.exit(1)
    else:
        print("\nAll CAN interfaces configured successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
