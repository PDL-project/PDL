# Task Description: Put all groceries in the fridge

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analyze due to previous subtask:
# 1. Robot not at apple location
# 2. Robot not holding apple
# 3. Fridge is initially closed

# SubTask 1: Put the Apple in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: Apple(Location=CounterTop), Fridge(Location=Floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not at bread location
# 2. Robot not holding bread
# 3. Fridge is initially closed

# SubTask 2: Put the Bread in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: Bread(Location=CounterTop), Fridge(Location=Floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not at lettuce location
# 2. Robot not holding lettuce
# 3. Fridge is initially closed

# SubTask 3: Put the Lettuce in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: Lettuce(Location=CounterTop), Fridge(Location=Floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not at tomato location
# 2. Robot not holding tomato
# 3. Fridge is initially closed

# SubTask 4: Put the Tomato in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: Tomato(Location=CounterTop), Fridge(Location=Floor)

# Task Put all groceries in the fridge is done.