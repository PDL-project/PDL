# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION
# Decompose and parallelize subtasks where ever possible
# Independent subtasks:

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding fork.
# 2. Robot not at fork location.

# SubTask 1: Move the Fork to the Drawer.
    Skills Required: GoToObject, PickupObject, GoToObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=diningTable), drawer(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding spoon.
# 2. Robot not at spoon location.

# SubTask 2: Move the Spoon to the Drawer.
    Skills Required: GoToObject, PickupObject, GoToObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=diningTable), drawer(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding mug.
# 2. Robot not at mug location.

# SubTask 3: Move the Mug to the CoffeeMachine.
    Skills Required: GoToObject, PickupObject, GoToObject, PutObject
    Related Objects: mug(Location=diningTable), coffeeMachine(Location=diningTable)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding lettuce.
# 2. Robot not at lettuce location.

# SubTask 4: Move the Lettuce to the Fridge.
    Skills Required: GoToObject, PickupObject, GoToObject, OpenObject, PutObject, CloseObject
    Related Objects: lettuce(Location=diningTable), fridge(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding tomato.
# 2. Robot not at tomato location.

# SubTask 5: Move the Tomato to the Fridge.
    Skills Required: GoToObject, PickupObject, GoToObject, OpenObject, PutObject, CloseObject
    Related Objects: tomato(Location=diningTable), fridge(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding toaster.
# 2. Robot not at toaster location.

# SubTask 6: Move the Toaster to the CounterTop.
    Skills Required: GoToObject, PickupObject, GoToObject, PutObject
    Related Objects: toaster(Location=diningTable), counterTop(Location=floor)

# Task Clear the table by placing items at their appropriate positions is done.