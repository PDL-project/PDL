# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding SaltShaker.
# 2. Robot not at SaltShaker location.
# 3. Fridge is initially closed.

# SubTask 1: Put the SaltShaker in the Fridge. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: SaltShaker(Location=CounterTop), Fridge(Location=Floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding PepperShaker.
# 2. Robot not at PepperShaker location.
# 3. Fridge is initially closed.

# SubTask 2: Put the PepperShaker in the Fridge. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: PepperShaker(Location=CounterTop), Fridge(Location=Floor)

# Task Put all shakers in the fridge is done.