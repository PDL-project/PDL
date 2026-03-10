(define (problem open_drawer_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    drawer2 - object
    counterTop - object
  )
  (:init
    (at robot2 counterTop)
    (at-location drawer2 counterTop)
  )
  (:goal
    (and
      (object-open robot2 drawer2)
    )
  )
)