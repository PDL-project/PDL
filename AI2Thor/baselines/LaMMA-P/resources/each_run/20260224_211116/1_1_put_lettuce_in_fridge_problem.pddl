(define (problem put_lettuce_in_fridge_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    lettuce - object
    fridge - object
    counterTop - object
  )
  (:init
    (at robot2 counterTop)
    (at-location lettuce counterTop)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location lettuce fridge)
    )
  )
)