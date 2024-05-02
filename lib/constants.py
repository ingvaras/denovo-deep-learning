from enum import Enum

LEARNING_RATE = 0.0025
L1 = 6e-5
POSITIVE_NEGATIVE_RATIO = 0.2


class MutationType(Enum):
    Insertion = 'insertion'
    Deletion = 'deletion'
    Substitution = 'substitution'
