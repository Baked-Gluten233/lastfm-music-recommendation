"""Runtime invariants, including when Python is invoked with -O."""

def require(condition, message):
    if not condition:
        raise ValueError(message)
