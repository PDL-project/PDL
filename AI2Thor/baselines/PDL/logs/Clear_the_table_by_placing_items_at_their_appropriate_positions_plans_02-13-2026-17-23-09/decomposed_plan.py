# Task Description: Clear the table by placing items at their appropriate positions

# GENERAL TASK DECOMPOSITION
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analysis:
# 1. Robot not at any object location
# 2. Robot not holding any object

# SubTask 1: Move the Bread to the CounterTop.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: bread(Location=DiningTable), counterTop(Location=Floor)

# Initial condition analysis due to previous subtask:
# 1. Robot not at fork location
# 2. Robot not holding fork

# SubTask 2: Move the Fork to the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=DiningTable), drawer(Location=Floor)

# Initial condition analysis due to previous subtask:
# 1. Robot not at spoon location
# 2. Robot not holding spoon

# SubTask 3: Move the Spoon to the Drawer.
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=DiningTable), drawer(Location=Floor)

# Initial condition analysis due to previous subtask:
# 1. Robot not at mug location
# 2. Robot not holding mug

# SubTask 4: Move the Mug to the CoffeeMachine.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: mug(Location=DiningTable), coffeeMachine(Location=DiningTable)

# Initial condition analysis due to previous subtask:
# 1. Robot not at pan location
# 2. Robot not holding pan

# SubTask 5: Move the Pan to the StoveBurner.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: pan(Location=DiningTable), stoveBurner(Location=Floor)

# Initial condition analysis due to previous subtask:
# 1. Robot not at tomato location
# 2. Robot not holding tomato

# SubTask 6: Move the Tomato to the CounterTop.
    Skills Required: GoToObject, PickupObject, PutObject
    Related Objects: tomato(Location=DiningTable), counterTop(Location=Floor)

# Task Clear the table by placing items at their appropriate positions is done.