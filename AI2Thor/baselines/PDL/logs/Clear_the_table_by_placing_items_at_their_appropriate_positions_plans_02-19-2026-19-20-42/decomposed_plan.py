# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible

# Initial condition analyze due to previous subtask:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 1: Move the Bowl to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: bowl(Location=diningTable), appropriatePosition(Location=counterTop)

# SubTask 2: Move the Bread to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: bread(Location=diningTable), appropriatePosition(Location=counterTop)

# SubTask 3: Move the CoffeeMachine to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: coffeeMachine(Location=diningTable), appropriatePosition(Location=counterTop)

# SubTask 4: Move the Fork to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: fork(Location=diningTable), appropriatePosition(Location=drawer)

# SubTask 5: Move the Mug to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=diningTable), appropriatePosition(Location=cabinet)

# SubTask 6: Move the Pan to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: pan(Location=diningTable), appropriatePosition(Location=stoveBurner)

# SubTask 7: Move the Spoon to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: spoon(Location=diningTable), appropriatePosition(Location=drawer)

# SubTask 8: Move the Toaster to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: toaster(Location=diningTable), appropriatePosition(Location=counterTop)

# SubTask 9: Move the Tomato to its appropriate position.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: tomato(Location=diningTable), appropriatePosition(Location=fridge)

# Task Clear the table by placing items at their appropriate positions is done.