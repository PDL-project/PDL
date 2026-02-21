# Task Description: Put all groceries in the fridge

# Initial condition analysis:
# 1. The fridge is initially closed.
# 2. The robot is not holding any groceries.
# 3. Groceries are on the countertop.

# Multi-instance grounding rule:
# - Fridge x1 -> fridge1
# - Apple x1 -> apple1
# - Bread x1 -> bread1
# - Lettuce x1 -> lettuce1
# - Tomato x1 -> tomato1

# GENERAL TASK DECOMPOSITION
# Decompose and parallelize subtasks where possible.

# SubTask 1: Open fridge1.
    Skills Required: GoToObject, OpenObject
    Related Objects: fridge1(Location=floor)

# SubTask 2: Put apple1 into fridge1.
    Skills Required: GoToObject, PickupObject, PutObjectInFridge
    Related Objects: apple1(Location=CounterTop), fridge1(Location=floor)

# SubTask 3: Put bread1 into fridge1.
    Skills Required: GoToObject, PickupObject, PutObjectInFridge
    Related Objects: bread1(Location=CounterTop), fridge1(Location=floor)

# SubTask 4: Put lettuce1 into fridge1.
    Skills Required: GoToObject, PickupObject, PutObjectInFridge
    Related Objects: lettuce1(Location=CounterTop), fridge1(Location=floor)

# SubTask 5: Put tomato1 into fridge1.
    Skills Required: GoToObject, PickupObject, PutObjectInFridge
    Related Objects: tomato1(Location=CounterTop), fridge1(Location=floor)

# SubTask 6: Close fridge1.
    Skills Required: GoToObject, CloseObject
    Related Objects: fridge1(Location=floor)

# Task Put all groceries in the fridge is done.