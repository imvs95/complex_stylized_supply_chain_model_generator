def find_overlaps(left_tuple: list, right_tuple: list):
    """Function that checks whether two tuple ranges overlap.

    Args:
        left_tuple (list): list with two values forming a range
        right_tuple (list): list with two values forming a range

    Returns:
        bool: if there is an overlap returns True. Else returns False.
    """

    if min(left_tuple[1], right_tuple[1]) != max(left_tuple[0], right_tuple[0]):
        if (
            max(
                0,
                min(left_tuple[1], right_tuple[1]) - max(left_tuple[0], right_tuple[0]),
            )
            > 0
        ):
            return True
        else:
            return False
    else:
        return True
