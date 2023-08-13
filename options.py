from enum import Enum, auto

class StemmingOptions(Enum):
    NONE = auto()
    ISRILIGHT = auto()
    ISRI = auto()
    ARABICLIGHT = auto()

class StopWordsOptions(Enum):
    KEEP = auto()
    REMOVE = auto()
