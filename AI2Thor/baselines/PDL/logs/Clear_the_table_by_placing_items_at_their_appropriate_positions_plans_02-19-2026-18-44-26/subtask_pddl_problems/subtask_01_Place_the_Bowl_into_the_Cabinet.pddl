```lisp
(define (problem place-bowl-in-cabinet)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bowl - object
    cabinet1 - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bowl diningtable)
    (at-location cabinet1 floor)
    (object-close robot1 cabinet1)
    (not (holding robot1 bowl))
  )

  (:goal (and
    (at-location bowl cabinet1)
    (object-close robot1 cabinet1)
  ))
)
```