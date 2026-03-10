(define (problem put_potato_in_fridge_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    potato - object
    fridge - object
    counterTop - object
    kitchenArea - object
  )
  (:init
    (at robot2 counterTop)
    (at-location potato counterTop)
    (at-location fridge kitchenArea)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location potato fridge)
    )
  )
)