# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analyze due to previous subtask:
# 1. Robot not at bread location
# 2. Robot not holding bread
# 3. Fridge is initially closed

# SubTask 1: Put the Bread in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: bread(Location=counterTop), fridge(Location=floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not at lettuce location
# 2. Robot not holding lettuce
# 3. Fridge is initially closed

# SubTask 2: Put the Lettuce in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: lettuce(Location=counterTop), fridge(Location=floor)

# Initial condition analyze due to previous subtask:
# 1. Robot not at tomato location
# 2. Robot not holding tomato
# 3. Fridge is initially closed

# SubTask 3: Put the Tomato in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: tomato(Location=counterTop), fridge(Location=floor)

# Task Put the bread, lettuce, and tomato in the fridge is done.