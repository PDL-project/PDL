# Task Description: Using the appropriate resources, extinguish the chemical fire CaldorFire and
# the non-chemical fire GreatFire. Also, explore to find LostPersonTimmy, carry them,
# and drop them in a deposit.

# GENERAL TASK DECOMPOSITION
# Decompose and parallelize subtasks where ever possible
# Independent subtasks: SubTask 1 and SubTask 2 can run in parallel;
#                       SubTask 3 can also run in parallel with 1 and 2.

# Initial condition analyze due to previous subtask:
# 1. Robot not at any reservoir
# 2. Robot not holding any supply
# 3. CaldorFire regions are active (on fire)

# SubTask 1: Extinguish CaldorFire using Sand.
    Skills Required: GoToObject, GetSupply, UseSupply
    Related Objects: ReservoirUtah(type=Sand), CaldorFire_Region_1(fire=CaldorFire), CaldorFire_Region_2(fire=CaldorFire)

# Initial condition analyze due to previous subtask:
# 1. Robot not at any reservoir
# 2. Robot not holding any supply
# 3. GreatFire regions are active (on fire)

# SubTask 2: Extinguish GreatFire using Water.
    Skills Required: GoToObject, GetSupply, UseSupply
    Related Objects: ReservoirYork(type=Water), GreatFire_Region_1(fire=GreatFire)

# Initial condition analyze due to previous subtask:
# 1. Robot has not explored yet
# 2. LostPersonTimmy position unknown
# 3. Robot not carrying anyone

# SubTask 3: Find and rescue LostPersonTimmy. [Multi-Agent: 2 robots required for Carry+DropOff]
    Skills Required: Explore, GoToObject, Carry, DropOff
    Related Objects: LostPersonTimmy(type=Person), DepositFacility(type=Deposit)
    Note: Moving a person requires 2 agents collaborating simultaneously.

# Task is done.


# Task Description: Using the appropriate resources, extinguish the chemical fire BlueFire.
# Also, explore to find LostPersonAlice and LostPersonBob, carry them, and drop them in a deposit.

# GENERAL TASK DECOMPOSITION
# Decompose and parallelize subtasks where ever possible
# Independent subtasks: SubTask 1 and SubTask 2 can run in parallel.

# Initial condition analyze due to previous subtask:
# 1. Robot not at any reservoir
# 2. Robot not holding Sand
# 3. BlueFire_Region_1 is active

# SubTask 1: Extinguish BlueFire using Sand.
    Skills Required: GoToObject, GetSupply, UseSupply
    Related Objects: ReservoirAlpha(type=Sand), BlueFire_Region_1(fire=BlueFire), BlueFire_Region_2(fire=BlueFire)

# Initial condition analyze due to previous subtask:
# 1. Robot not explored
# 2. LostPersonAlice and LostPersonBob locations unknown
# 3. Robot not carrying anyone

# SubTask 2: Find and rescue LostPersonAlice. [Multi-Agent: 2 robots required for Carry+DropOff]
    Skills Required: Explore, GoToObject, Carry, DropOff
    Related Objects: LostPersonAlice(type=Person), SafeZone(type=Deposit)
    Note: Moving a person requires 2 agents collaborating simultaneously.

# SubTask 3: Find and rescue LostPersonBob. [Multi-Agent: 2 robots required for Carry+DropOff]
    Skills Required: Explore, GoToObject, Carry, DropOff
    Related Objects: LostPersonBob(type=Person), SafeZone(type=Deposit)
    Note: Moving a person requires 2 agents collaborating simultaneously.

# Task is done.
