# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analysis:
# 1. Robot not at salt shaker location.
# 2. Robot not holding salt shaker.
# 3. Fridge is initially closed.

# SubTask 1: Put the Salt Shaker in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: SaltShaker(Location=CounterTop), Fridge(Location=Floor)

# Initial condition analysis:
# 1. Robot not at pepper shaker location.
# 2. Robot not holding pepper shaker.
# 3. Fridge is initially closed.

# SubTask 2: Put the Pepper Shaker in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: PepperShaker(Location=CounterTop), Fridge(Location=Floor)

# Task Put all shakers in the fridge is done.