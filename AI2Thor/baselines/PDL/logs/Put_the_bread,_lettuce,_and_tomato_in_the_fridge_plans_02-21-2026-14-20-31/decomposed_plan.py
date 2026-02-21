# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible.

# Initial condition analyze due to previous subtask:
# 1. Robot not at any object location
# 2. Robot not holding any object
# 3. Fridge is initially closed

# SubTask 1: Put the Bread in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: Bread(Location=CounterTop), Fridge(Location=Floor)

# SubTask 2: Put the Lettuce in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: Lettuce(Location=CounterTop), Fridge(Location=Floor)

# SubTask 3: Put the Tomato in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: Tomato(Location=CounterTop), Fridge(Location=Floor)

# Task Put the bread, lettuce, and tomato in the fridge is done.