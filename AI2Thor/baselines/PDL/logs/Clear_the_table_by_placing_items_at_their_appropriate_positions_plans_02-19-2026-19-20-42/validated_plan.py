(define (problem move-bowl-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bowl - object
    diningtable - object
    countertop - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bowl diningtable)
    (not (holding robot1 bowl))
  )

  (:goal (and
    (at-location bowl countertop)
    (not (holding robot1 bowl))
  ))
)