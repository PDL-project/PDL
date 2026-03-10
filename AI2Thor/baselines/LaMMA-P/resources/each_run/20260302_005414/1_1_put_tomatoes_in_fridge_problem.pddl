(define (problem put_tomatoes_in_fridge_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    tomato - object
    fridge - object
    counterTop - object
    kitchenArea - object
  )
  (:init
    (at robot1 counterTop)
    (at-location tomato counterTop)
    (at-location fridge kitchenArea)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location tomato fridge)
    )
  )
)