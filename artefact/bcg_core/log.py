# SPDX-License-Identifier: GPL-3.0-or-later
"""One logging setup for all apps (modules only call ``logging.getLogger(__name__)``)."""
import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once; later calls do nothing."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
