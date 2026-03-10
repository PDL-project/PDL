(define (problem put_mug_in_fridge_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    mug - object
    fridge - object
    counterTop - object
    kitchenArea - object
  )
  (:init
    (at robot2 counterTop)
    (at-location mug counterTop)
    (at-location fridge kitchenArea)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location mug fridge)
    )
  )
)