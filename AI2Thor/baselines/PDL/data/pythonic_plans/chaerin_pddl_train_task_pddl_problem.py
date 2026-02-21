# EXAMPLE 1 - Task Description: Turn off the light

# pddl problem file
(define (problem switch-off-light)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    lightswitch - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location lightswitch floor)
    (switch-on robot1 lightswitch)
  )

  (:goal (and
    (switch-off robot1 lightswitch)
  ))
)


# EXAMPLE 2 - Task Description: Slice the Potato 

# problem file
(define (problem slice-potato)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    potato - object 
    knife - object
    cuttingboard - object
    diningtable - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))

    (at robot1 kitchen)

    (at-location potato diningtable)
    (at-location knife diningtable)
    (at-location cuttingboard diningtable)

    (not (holding robot1 potato))
    (not (holding robot1 knife))
  )

  (:goal (and
    (sliced potato)
  ))
)


# EXAMPLE 3 - Task Description: Put the Apple in the Fridge

# problem file
(define (problem put-apple-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    apple - object
    fridge - object
    kitchen - object
    countertop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location apple countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 apple))
  )

  (:goal (and
    (at-location apple fridge)
    (not (holding robot1 apple))
  ))
)


# EXAMPLE 4 - Task Description: Open first Drawer
# IMPORTANT MULTI-INSTANCE RULE:
# If ENVIRONMENT OBJECTS contains multiple objects of the same type (e.g., 3 Drawers),
# you MUST enumerate them as drawer1, drawer2, drawer3 in :objects and use each in goals.
#
# Example grounding from ENVIRONMENT OBJECTS:
# Drawers found = 3  -> drawer1, drawer2, drawer3

# problem file
(define (problem open-first-drawers)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer1 - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 floor)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
  )

  (:goal (and
    (object-open robot1 drawer1)
  ))
)
