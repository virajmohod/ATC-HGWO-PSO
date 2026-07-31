"""
=========================================================
Follower Salp Update
=========================================================
"""

from __future__ import annotations

import numpy as np


class FollowerUpdate:

    @staticmethod
    def update(previous, current):

        return 0.5 * (previous + current)