(define (problem place-bowl-in-cabinet)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bowl - object
    cabinet - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bowl diningtable)
    (at-location cabinet floor)
    (object-close robot1 cabinet)
    (not (holding robot1 bowl))
  )

  (:goal (and
    (at-location bowl cabinet)
    (object-close robot1 cabinet)
  ))
)