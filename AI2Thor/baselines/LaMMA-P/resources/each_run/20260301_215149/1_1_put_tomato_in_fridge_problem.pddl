(define (problem put_tomato_in_fridge_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    tomato - object
    fridge - object
    counterTop - object
    floor - object
  )
  (:init
    (at robot3 counterTop)
    (at-location tomato counterTop)
    (at-location fridge floor)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location tomato fridge)
    )
  )
)