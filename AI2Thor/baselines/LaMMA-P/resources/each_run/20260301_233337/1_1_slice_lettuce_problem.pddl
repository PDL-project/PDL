(define (problem slice_lettuce_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    lettuce - object
    counterTop - object
  )
  (:init
    (at robot2 counterTop)
    (at-location lettuce counterTop)
  )
  (:goal
    (and
      (sliced lettuce)
    )
  )
)