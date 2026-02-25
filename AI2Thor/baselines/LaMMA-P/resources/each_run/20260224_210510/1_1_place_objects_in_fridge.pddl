(define (problem place_objects_in_fridge)
  (:domain robot1)
  (:objects
    robot1 - robot
    robot2 - robot
    robot3 - robot
    bread - object
    lettuce - object
    tomato - object
    fridge - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at robot2 counterTop)
    (at robot3 counterTop)
    (at-location bread counterTop)
    (at-location lettuce counterTop)
    (at-location tomato counterTop)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location bread fridge)
      (at-location lettuce fridge)
      (at-location tomato fridge)
    )
  )
)