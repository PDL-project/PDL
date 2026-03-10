"""
To check completion of subtasks in 4_slice_all_sliceable task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Bread)
    • SliceObject(Bread)
    • NavigateTo(Tomato)
    • SliceObject(Tomato)
    • NavigateTo(Lettuce)
    • SliceObject(Lettuce)
    • NavigateTo(Egg)
    • SliceObject(Egg)
    • NavigateTo(Apple)
    • SliceObject(Apple)
    • NavigateTo(Potato)
    • SliceObject(Potato)
Coverage:
    • Bread
    • Tomato
    • Lettuce
    • Egg
    • Apple
    • Potato
"""

from AI2Thor.baselines.utils.checker import BaseChecker


class Checker(BaseChecker):
    def __init__(self) -> None:
        subtasks = [
            "NavigateTo(Bread)",
            "SliceObject(Bread)",
            "NavigateTo(Tomato)",
            "SliceObject(Tomato)",
            "NavigateTo(Lettuce)",
            "SliceObject(Lettuce)",
            "NavigateTo(Egg)",
            "SliceObject(Egg)",
            "NavigateTo(Apple)",
            "SliceObject(Apple)",
            "NavigateTo(Potato)",
            "SliceObject(Potato)",
        ]
        conditional_subtasks = []
        independent_subtasks = [
            "NavigateTo(Bread)",
            "SliceObject(Bread)",
            "NavigateTo(Tomato)",
            "SliceObject(Tomato)",
            "NavigateTo(Lettuce)",
            "SliceObject(Lettuce)",
            "NavigateTo(Egg)",
            "SliceObject(Egg)",
            "NavigateTo(Apple)",
            "SliceObject(Apple)",
            "NavigateTo(Potato)",
            "SliceObject(Potato)",
        ]
        coverage = ["Bread", "Tomato", "Lettuce", "Egg", "Apple", "Potato"]
        interact_objects = ["Bread", "Tomato", "Lettuce", "Egg", "Apple", "Potato"]
        interact_receptacles = []

        super().__init__(
            subtasks,
            conditional_subtasks,
            independent_subtasks,
            coverage,
            interact_objects,
            interact_receptacles,
        )
