"""Shared networking utilities for FlowSim scripts and tests."""

import socket
import time


def wait_for_port(host: str, port: int, timeout: int = 600) -> bool:
    """Block until *host:port* accepts a TCP connection.

    Parameters
    ----------
    host : str
        Hostname or IP address to connect to.
    port : int
        Port number.
    timeout : int
        Maximum seconds to wait before returning False.

    Returns
    -------
    bool
        True if the port became reachable within *timeout*, False otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            time.sleep(2)
    return False
