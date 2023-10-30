__all__ = ["FROM_CSV", "FROM_NBA_STATS", "_PATH"]

# Macros and definitions
_PATH: str = "./resources/"

# flags
FROM_CSV = 1 << 0
FROM_NBA_STATS = 1 << 1
