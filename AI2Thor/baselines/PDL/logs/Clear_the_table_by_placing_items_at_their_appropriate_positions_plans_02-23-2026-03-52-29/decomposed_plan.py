# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analyze due to previous subtask:
#1. Robot not at any object location
#2. Robot not holding any object

# SubTask 1: Move the Fork to the appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: fork(Location=diningTable), appropriatePosition(Location=counterTop)

# Initial condition analyze due to previous subtask:
#1. Robot not at spoon location
#2. Robot not holding spoon

# SubTask 2: Move the Spoon to the appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: spoon(Location=diningTable), appropriatePosition(Location=counterTop)

# Initial condition analyze due to previous subtask:
#1. Robot not at mug location
#2. Robot not holding mug

# SubTask 3: Move the Mug to the appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=diningTable), appropriatePosition(Location=coffeeMachine)

# Initial condition analyze due to previous subtask:
#1. Robot not at lettuce location
#2. Robot not holding lettuce

# SubTask 4: Move the Lettuce to the appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: lettuce(Location=diningTable), appropriatePosition(Location=fridge)

# Initial condition analyze due to previous subtask:
#1. Robot not at tomato location
#2. Robot not holding tomato

# SubTask 5: Move the Tomato to the appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: tomato(Location=diningTable), appropriatePosition(Location=fridge)

# Initial condition analyze due to previous subtask:
#1. Robot not at toaster location
#2. Robot not holding toaster

# SubTask 6: Move the Toaster to the appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: toaster(Location=diningTable), appropriatePosition(Location=counterTop)

# Task Clear the table by placing items at their appropriate positions is done.