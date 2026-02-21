# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible.

# Initial condition analysis:
# 1. Robot not at any object location.
# 2. Robot not holding any object.
# 3. Cabinet, Drawer, and Fridge are initially closed.

# SubTask 1: Place the Bowl into the Cabinet.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: Bowl(Location=DiningTable), Cabinet(Location=Floor)

# SubTask 2: Place the Bread into the Cabinet.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: Bread(Location=DiningTable), Cabinet(Location=Floor)

# SubTask 3: Place the Fork into the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: Fork(Location=DiningTable), Drawer(Location=Floor)

# SubTask 4: Place the Spoon into the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: Spoon(Location=DiningTable), Drawer(Location=Floor)

# SubTask 5: Place the Mug into the Cabinet.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: Mug(Location=DiningTable), Cabinet(Location=Floor)

# SubTask 6: Place the Pan into the Cabinet.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: Pan(Location=DiningTable), Cabinet(Location=Floor)

# SubTask 7: Place the Tomato into the Fridge.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: Tomato(Location=DiningTable), Fridge(Location=Floor)

# Task Clear the table by placing items at their appropriate positions is done.