(define (problem wash_bowl_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    bowl - object
    sink - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at-location bowl counterTop)
    (at-location sink counterTop) ; Assuming sink is at counterTop for this example
    (inaction robot1) ; Assuming the robot starts in an inactive state
  )
  (:goal
    (and
      (cleaned robot1 bowl)
    )
  )
)