"""
aided.core.utils

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from typing import List


def split_work(n: int, num_groups: int) -> List[List[int]]:
    """Divide n units into num_groups groups for work distribution.

    Args:
        n: Number of units to divide.
        num_groups: Number of groups to divide into.

    Returns:
        groups: List of lists of units for each group.
    """
    if n == 0:
        return []

    group_size = (n + num_groups - 1) // num_groups
    groups = [list(range(i, min(i + group_size, n))) for i in range(0, n, group_size)]
    return groups
