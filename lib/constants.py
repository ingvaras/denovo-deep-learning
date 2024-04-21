from enum import Enum

LEARNING_RATE = 3e-4


class MutationType(Enum):
    Insertion = 'insertion'
    Deletion = 'deletion'
    Substitution = 'substitution'
