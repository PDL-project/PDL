(define (problem move-bowl-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bowl - object
    diningTable - object
    counterTop - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bowl diningTable)
    (not (holding robot1 bowl))
  )

  (:goal (and
    (at-location bowl counterTop)
    (not (holding robot1 bowl))
  ))
)