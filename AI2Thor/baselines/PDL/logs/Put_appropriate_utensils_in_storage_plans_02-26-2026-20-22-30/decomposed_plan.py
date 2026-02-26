# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding fork.
# 2. Robot not at fork location.

# SubTask 1: Store the Fork. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: fork(Location=counterTop), drawer(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding spoon.
# 2. Robot not at spoon location.

# SubTask 2: Store the Spoon. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: spoon(Location=counterTop), drawer(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding knife.
# 2. Robot not at knife location.

# SubTask 3: Store the Knife. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: knife(Location=counterTop), drawer(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding butter knife.
# 2. Robot not at butter knife location.

# SubTask 4: Store the ButterKnife. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: butterKnife(Location=counterTop), drawer(Location=floor)

# Initial Precondition analyze due to previous subtask:
# 1. Robot not holding spatula.
# 2. Robot not at spatula location.

# SubTask 5: Store the Spatula. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: spatula(Location=counterTop), drawer(Location=floor)

# Task Put appropriate utensils in storage is done.