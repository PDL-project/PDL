(define (problem put_bowl_in_fridge_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    bowl - object
    fridge - object
    counterTop - object
  )
  (:init
    (at robot3 counterTop)
    (at-location bowl counterTop)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location bowl fridge)
    )
  )
)