# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analyze due to previous subtask:
# 1. Robot not holding any object.
# 2. Robot not at any specific object location.

# SubTask 1: Move the Fork to the Drawer.
    Skills Required: GoToObject, PickupObject, GoToObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=diningTable), drawer(Location=floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not holding any object.
# 2. Robot not at any specific object location.

# SubTask 2: Move the Spoon to the Drawer.
    Skills Required: GoToObject, PickupObject, GoToObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=diningTable), drawer(Location=floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not holding any object.
# 2. Robot not at any specific object location.

# SubTask 3: Move the Mug to the CoffeeMachine.
    Skills Required: GoToObject, PickupObject, GoToObject, PutObject
    Related Objects: mug(Location=diningTable), coffeeMachine(Location=diningTable)

# Initial condition analyze due to previous subtask:
# 1. Robot not holding any object.
# 2. Robot not at any specific object location.

# SubTask 4: Move the Lettuce to the Fridge.
    Skills Required: GoToObject, PickupObject, GoToObject, OpenObject, PutObject, CloseObject
    Related Objects: lettuce(Location=diningTable), fridge(Location=floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not holding any object.
# 2. Robot not at any specific object location.

# SubTask 5: Move the Tomato to the Fridge.
    Skills Required: GoToObject, PickupObject, GoToObject, OpenObject, PutObject, CloseObject
    Related Objects: tomato(Location=diningTable), fridge(Location=floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not holding any object.
# 2. Robot not at any specific object location.

# SubTask 6: Move the Toaster to its original position on the CounterTop.
    Skills Required: GoToObject, PickupObject, GoToObject, PutObject
    Related Objects: toaster(Location=diningTable), counterTop(Location=floor)

# Task Clear the table by placing items at their appropriate positions is done.