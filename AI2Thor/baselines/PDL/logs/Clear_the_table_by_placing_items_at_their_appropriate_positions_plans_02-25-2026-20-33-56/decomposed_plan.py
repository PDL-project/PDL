# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 1: Place the Fork in the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=diningTable), drawer(Location=floor)

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 2: Place the Spoon in the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=diningTable), drawer(Location=floor)

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 3: Place the Mug on the CoffeeMachine.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=diningTable), coffeeMachine(Location=diningTable)

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 4: Place the Lettuce in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: lettuce(Location=diningTable), fridge(Location=floor)

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 5: Place the Tomato in the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObjectInFridge, CloseObject
    Related Objects: tomato(Location=diningTable), fridge(Location=floor)

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 6: Place the Toaster on the CounterTop.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: toaster(Location=diningTable), counterTop(Location=floor)

# Task Clear the table by placing items at their appropriate positions is done.