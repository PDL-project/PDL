# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 1: Move the Fork to the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=diningTable), drawer(Location=floor)

# SubTask 2: Move the Spoon to the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=diningTable), drawer(Location=floor)

# SubTask 3: Move the Mug to the CoffeeMachine.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=diningTable), coffeeMachine(Location=diningTable)

# SubTask 4: Move the Lettuce to the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: lettuce(Location=diningTable), fridge(Location=floor)

# SubTask 5: Move the Tomato to the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: tomato(Location=diningTable), fridge(Location=floor)

# SubTask 6: Move the Toaster to the CounterTop.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: toaster(Location=diningTable), counterTop(Location=floor)

# Task Clear the table by placing items at their appropriate positions is done.