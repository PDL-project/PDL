(define (problem open_drawer_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    drawer1 - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at-location drawer1 counterTop)
  )
  (:goal
    (and
      (object-open robot1 drawer1)
    )
  )
)