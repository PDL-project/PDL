(define (problem move-bowl-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bowl - object
    diningTable - object
    counterTop - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bowl diningTable)
    (not (holding robot1 bowl))
  )

  (:goal (and
    (at-location bowl counterTop)
    (not (holding robot1 bowl))
  ))

  (:metric minimize (total-cost))
)