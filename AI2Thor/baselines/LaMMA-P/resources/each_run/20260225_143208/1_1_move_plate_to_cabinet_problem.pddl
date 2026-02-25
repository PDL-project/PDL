(define (problem move_plate_to_cabinet_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    plate - object
    cabinet - object
    diningTable - object
    counterTop - object
  )
  (:init
    (at robot1 diningTable)
    (at-location plate diningTable)
    (at-location cabinet counterTop)
  )
  (:goal
    (and
      (at-location plate cabinet)
    )
  )
)