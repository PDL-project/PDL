(define (problem open_drawer4_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    drawer4 - object
  )
  (:init
    (at robot1 counterTop) ; Assuming initial location of the robot is counterTop
    (at-location drawer4 counterTop) ; Assuming initial location of drawer4 is counterTop
  )
  (:goal
    (and
      (object-open robot1 drawer4)
    )
  )
)