# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:

# Initial condition analyze due to previous subtask:
#1. Robot at fridge 1
#2. Robot holding egg
#3. fridge 1 is opened
#4. fridge 1 is already full 

# SubTask 1: Put an Egg in the Fridge2. 
    Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject
    Related Objects: egg(Location=fridge), fridge2(Location=floor)

GoToObject(robot1, fridge2)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 fridge2)

OpenFridge(robot1, fridge2)
   Preconditions:
     (not (inaction robot1))
     (at robot1 fridge2)
     (is-fridge fridge2)
   Effects:
     (object-open robot1 fridge2)
     (fridge-open fridge2)

PutObjectInFridge(robot1, egg, fridge2)
   Preconditions:
     (holding robot1 egg)
     (at robot1 fridge2)
     (not (inaction robot1))
     (is-fridge fridge2)
     (fridge-open fridge2)
   Effects:
     (at-location egg fridge2)
     (not (holding robot1 egg))
     (not (inaction robot1))

CloseFridge(robot1, fridge2)
   Preconditions:
     (not (inaction robot1))
     (at robot1 fridge2)
     (object-open robot1 fridge2)
     (is-fridge fridge2)
   Effects:
     (object-close robot1 fridge2)
     (not (fridge-open fridge2))

Goal condition: 
(at-location egg fridge2)

# Initial condition analyze due to previous subtask:
#1. Robot not at apple location
#2. Robot not holding apple
#3. Robot not holding knife
#4. knife can't put in drawer1. 

# SubTask 2: Prepare Apple Slices. 
    Skills Required: GoToObject, PickupObject, SliceObject, PutObject
    Related Objects: apple(Location=counterTop), knife(Location=diningTable), pot(Location=stoveBurner)

GoToObject(robot1, apple)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 apple)

PickupObject(robot1, apple, counterTop)
   Preconditions:
     (at-location apple counterTop)
     (at robot1 apple)
     (not (inaction robot1))
   Effects:
     (holding robot1 apple)

GoToObject(robot1, cuttingboard)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 cuttingboard)

SliceObject(robot1, apple, cuttingboard)
   Preconditions:
     (at-location apple cuttingboard)
     (at robot1 cuttingboard)
     (not (inaction robot1))
   Effects:
     (sliced apple)

GoToObject(robot1, pot)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 pot)

PutObject(robot1, apple, pot)
   Preconditions:
     (holding robot1 apple)
     (at robot1 pot)
     (not (inaction robot1))
   Effects:
     (at-location apple pot)
     (not (holding robot1 apple))

Goal condition:
(sliced apple)
(at-location apple pot)

# Inital condition analyze due to previous subtask:
#1. Robot at pot location
#2. Fridge2 is Fridge, and initally closed
#3. Robot not holding pot initally.

# SubTask 3: Place the Pot with Apple Slices in the Fridge.
    Skills Required: GoToObject, PickupObject, PutObject, OpenObject, CloseObject
    Related Objects: pot(Location=stoveBurner), apple(Location=counterTop), fridge2(Location=floor)

GoToObject(robot1, pot)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 pot)

PickupObject(robot1, pot, stoveBurner)
   Preconditions:
     (at-location pot stoveBurner)
     (at robot1 stoveBurner)
     (not (inaction robot1))
   Effects:
     (holding robot1 pot)

GoToObject(robot1, fridge2)
   Preconditions:
     (not (inaction robot1))
   Effects:
     (at robot1 fridge2)

OpenFridge(robot1, fridge2)
   Preconditions:
     (not (inaction robot1))
     (at robot1 fridge2)
     (is-fridge fridge2)
   Effects:
     (object-open robot1 fridge2)
     (fridge-open fridge2)

PutObjectInFridge(robot1, pot, fridge2)
   Preconditions:
     (holding robot1 pot)
     (at robot1 fridge2)
     (not (inaction robot1))
     (is-fridge fridge2)
     (fridge-open fridge2)
   Effects:
     (at-location pot fridge2)
     (not (holding robot1 pot))
     (not (inaction fridge2))

CloseFridge(robot1, fridge2)
   Preconditions:
     (not (inaction robot1))
     (at robot1 fridge2)
     (object-open robot1 fridge2)
     (is-fridge fridge2)
   Effects:
     (object-close robot1 fridge2)
     (not (fridge-open fridge2))

Goal condition:
(at-location pot fridge2)

# Task Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator is done.