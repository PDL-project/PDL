(define (problem move-bowl-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bowl - object
    countertop - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location bowl diningtable)
    (at-location countertop floor)
    (not (holding robot1 bowl))
  )

  (:goal (and
    (at-location bowl countertop)
    (not (holding robot1 bowl))
  ))

  (:metric minimize (total-cost))
)