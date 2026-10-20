"""
Shared constants used across training, merging, and evaluation.
All task-specific configurations are defined here once.
"""

# Total number of sequential tasks in the active forward sequence.
NUM_TASKS: int = 6

# Number of classes per task
NUM_CLASSES: list[int] = [2, 3, 2, 2, 2, 2]

# Total number of classes across all tasks
TOTAL_CLASSES: int = sum(NUM_CLASSES)  # = 13

# Embedding dimension from TITAN vision encoder
EMBED_DIM: int = 768

# Patch size used during WSI preprocessing
WSI_PATCH_PIXELS: int = 256

TITAN_PS_ARG: int = 1024

# Number of patches randomly sampled per slide during training/inference
K_PATCHES: int = 400

# Maps task_id → [start_class_idx, end_class_idx] in the global 13-class classifier
# Used to slice prompt prototype embeddings
TASK_CLASS_RANGES: dict[int, list[int]] = {
    0: [0, 1],    # BRCA: IDC(0), ILC(1)
    1: [2, 4],    # RCC:  CCRCC(2), PRCC(3), CHRCC(4)
    2: [5, 6],    # NSCLC: LUAD(5), LUSC(6)
    3: [7, 8],    # ESCA: class0(7), class1(8)
    4: [9, 10],   # TGCT: class0(9), class1(10)
    5: [11, 12],  # CESC: class0(11), class1(12)
}

# Maps task_id → maps local class idx → global class idx
# Used in Class-IL evaluation to convert per-task predictions to global space
TASK_TO_GLOBAL_CLASS: dict[int, dict[int, int]] = {
    0: {0: 0,  1: 1},
    1: {0: 2,  1: 3,  2: 4},
    2: {0: 5,  1: 6},
    3: {0: 7,  1: 8},
    4: {0: 9,  1: 10},
    5: {0: 11, 1: 12},
}

# Task names for logging
TASK_NAMES: list[str] = [
    "BRCA", "RCC", "NSCLC", "ESCA", "TGCT", "CESC",
]


# ---------------------------------------------------------------------------
# Forward-order aliases used alongside reverse-order constants.
# ---------------------------------------------------------------------------

TASK_NAMES_FORWARD:           list[str]                 = TASK_NAMES
NUM_CLASSES_FORWARD:          list[int]                 = NUM_CLASSES
TASK_CLASS_RANGES_FORWARD:    dict[int, list[int]]      = TASK_CLASS_RANGES
TASK_TO_GLOBAL_CLASS_FORWARD: dict[int, dict[int, int]] = TASK_TO_GLOBAL_CLASS

# The prompt classifier is always built in forward order for MLP initialization.
CLASSIFIER_CLASS_RANGES_FORWARD: dict[int, list[int]] = TASK_CLASS_RANGES_FORWARD


# ---------------------------------------------------------------------------
# REVERSE order: C→T→E→N→R→B
# ---------------------------------------------------------------------------

TASK_NAMES_REVERSE: list[str] = ["CESC", "TGCT", "ESCA", "NSCLC", "RCC", "BRCA"]

NUM_CLASSES_REVERSE: list[int] = [2, 2, 2, 2, 3, 2]

TASK_CLASS_RANGES_REVERSE: dict[int, list[int]] = {
    0: [0,  1],   # CESC: class0(0),  class1(1)
    1: [2,  3],   # TGCT: class0(2),  class1(3)
    2: [4,  5],   # ESCA: class0(4),  class1(5)
    3: [6,  7],   # NSCLC: class0(6), class1(7)
    4: [8,  10],  # RCC:  class0(8),  class1(9),  class2(10)
    5: [11, 12],  # BRCA: class0(11), class1(12)
}

TASK_TO_GLOBAL_CLASS_REVERSE: dict[int, dict[int, int]] = {
    0: {0: 0,  1: 1},
    1: {0: 2,  1: 3},
    2: {0: 4,  1: 5},
    3: {0: 6,  1: 7},
    4: {0: 8,  1: 9,  2: 10},
    5: {0: 11, 1: 12},
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_order_constants(order: str = "forward") -> tuple:
    """
    Return task metadata for the requested task order.

    Args:
        order: "forward" (B->R->N->E->T->C) or "reverse" (C->T->E->N->R->B).

    Returns:
        Tuple of task names, class counts, class ranges, and class mappings.

    Raises:
        ValueError: If order is not supported.
    """
    if order == "forward":
        return (
            TASK_NAMES_FORWARD,
            NUM_CLASSES_FORWARD,
            TASK_CLASS_RANGES_FORWARD,
            TASK_TO_GLOBAL_CLASS_FORWARD,
        )
    if order == "reverse":
        return (
            TASK_NAMES_REVERSE,
            NUM_CLASSES_REVERSE,
            TASK_CLASS_RANGES_REVERSE,
            TASK_TO_GLOBAL_CLASS_REVERSE,
        )
    raise ValueError(
        f"Unknown order '{order}'. Expected 'forward' or 'reverse'."
    )
