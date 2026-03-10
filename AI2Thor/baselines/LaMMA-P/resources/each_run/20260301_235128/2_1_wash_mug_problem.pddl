(define (problem wash_mug_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    mug - object
    counterTop - object
    sink - object
  )
  (:init
    (at robot2 counterTop)
    (at-location mug counterTop)
  )
  (:goal
    (and
      (cleaned robot2 mug)
    )
  )
)