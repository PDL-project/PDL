# GENERAL TASK DECOMPOSITION
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding fork.
# 2. Robot not at fork location.

# SubTask 1: Place the Fork in the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=diningTable), drawer(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding spoon.
# 2. Robot not at spoon location.

# SubTask 2: Place the Spoon in the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=diningTable), drawer(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding mug.
# 2. Robot not at mug location.

# SubTask 3: Place the Mug on the CoffeeMachine.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=diningTable), coffeeMachine(Location=diningTable)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding lettuce.
# 2. Robot not at lettuce location.

# SubTask 4: Place the Lettuce in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: lettuce(Location=diningTable), fridge(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding tomato.
# 2. Robot not at tomato location.

# SubTask 5: Place the Tomato in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: tomato(Location=diningTable), fridge(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding toaster.
# 2. Robot not at toaster location.

# SubTask 6: Place the Toaster on the CounterTop.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: toaster(Location=diningTable), counterTop(Location=floor)

# Task Clear the table by placing items at their appropriate positions is done.