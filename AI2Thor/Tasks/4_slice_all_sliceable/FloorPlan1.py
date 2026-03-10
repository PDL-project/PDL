"""
Pre-initialization for FloorPlan1 - Slice all sliceable objects.
Sliceable objects (Bread, Tomato, Lettuce, Egg, Apple, Potato) are already
accessible on countertops/tables in the default scene layout.
"""

class SceneInitializer:
    def __init__(self) -> None:
        pass

    def preinit(self, event, controller):
        """Pre-initialize the environment for the task.

        Args:
            event: env.event object
            controller: ai2thor.controller object

        Returns:
            event: env.event object
        """
        return event
