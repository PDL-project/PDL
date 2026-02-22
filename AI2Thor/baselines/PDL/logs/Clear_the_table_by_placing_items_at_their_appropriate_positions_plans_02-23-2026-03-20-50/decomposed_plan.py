# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 1: Move the Fork from the DiningTable to the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=DiningTable), drawer(Location=Floor)

# SubTask 2: Move the Spoon from the DiningTable to the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=DiningTable), drawer(Location=Floor)

# SubTask 3: Move the Mug from the DiningTable to the CoffeeMachine.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=DiningTable), coffeeMachine(Location=DiningTable)

# SubTask 4: Move the Lettuce from the DiningTable to the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: lettuce(Location=DiningTable), fridge(Location=Floor)

# SubTask 5: Move the Tomato from the DiningTable to the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: tomato(Location=DiningTable), fridge(Location=Floor)

# SubTask 6: Move the Toaster from the DiningTable to the CounterTop.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: toaster(Location=DiningTable), counterTop(Location=Floor)

# Task Clear the table by placing items at their appropriate positions is done.