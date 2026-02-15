(define (problem move-bowl-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bowl - object
    diningTable - object
    counterTop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 diningTable)
    (at-location bowl diningTable)
    (at-location counterTop floor)
    (not (holding robot1 bowl))
  )

  (:goal (and
    (at-location bowl counterTop)
    (not (holding robot1 bowl))
  ))
)