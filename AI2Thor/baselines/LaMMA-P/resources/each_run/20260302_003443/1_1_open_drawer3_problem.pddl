(define (problem open_drawer3_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    drawer3 - object
    counterTop - object
  )
  (:init
    (at robot3 counterTop)
    (at-location drawer3 counterTop)
  )
  (:goal
    (and
      (object-open robot3 drawer3)
    )
  )
)