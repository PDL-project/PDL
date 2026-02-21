# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analyze due to previous subtask:
#1. Robot not at any object location
#2. Robot not holding any object

# SubTask 1: Move the Fork to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: fork(Location=DiningTable), drawer(Location=Floor)

# SubTask 2: Move the Spoon to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: spoon(Location=DiningTable), drawer(Location=Floor)

# SubTask 3: Move the Mug to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=DiningTable), coffeeMachine(Location=DiningTable)

# SubTask 4: Move the Lettuce to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: lettuce(Location=DiningTable), fridge(Location=Floor)

# SubTask 5: Move the Tomato to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: tomato(Location=DiningTable), fridge(Location=Floor)

# SubTask 6: Move the Toaster to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: toaster(Location=DiningTable), counterTop(Location=Floor)

# Task Clear the table by placing items at their appropriate positions is done.