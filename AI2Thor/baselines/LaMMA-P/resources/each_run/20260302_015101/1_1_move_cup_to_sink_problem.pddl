(define (problem move_cup_to_sink_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    cup - object
    sink - object
    diningTable - object
    counterTop - object
  )
  (:init
    (at robot2 diningTable)
    (at-location cup diningTable)
    (at-location sink counterTop)
  )
  (:goal
    (and
      (at-location cup sink)
    )
  )
)