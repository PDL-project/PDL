# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analyze due to previous subtask:
#1. Robot not at any object location
#2. Robot not holding any object

# SubTask 1: Move the Fork from the DiningTable to the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=diningTable), drawer(Location=floor)

# Initial condition analyze due to previous subtask:
#1. Robot not at spoon location
#2. Robot not holding spoon

# SubTask 2: Move the Spoon from the DiningTable to the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=diningTable), drawer(Location=floor)

# Initial condition analyze due to previous subtask:
#1. Robot not at mug location
#2. Robot not holding mug

# SubTask 3: Move the Mug from the DiningTable to the CoffeeMachine.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=diningTable), coffeeMachine(Location=diningTable)

# Initial condition analyze due to previous subtask:
#1. Robot not at lettuce location
#2. Robot not holding lettuce

# SubTask 4: Move the Lettuce from the DiningTable to the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: lettuce(Location=diningTable), fridge(Location=floor)

# Initial condition analyze due to previous subtask:
#1. Robot not at tomato location
#2. Robot not holding tomato

# SubTask 5: Move the Tomato from the DiningTable to the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: tomato(Location=diningTable), fridge(Location=floor)

# Initial condition analyze due to previous subtask:
#1. Robot not at toaster location
#2. Robot not holding toaster

# SubTask 6: Move the Toaster from the DiningTable to the CounterTop.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: toaster(Location=diningTable), counterTop(Location=floor)

# Task Clear the table by placing items at their appropriate positions is done.