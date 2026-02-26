```lisp
(define (problem store-spatula)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spatula - object
    drawer1 - object
    counterTop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 floor)
    (at-location spatula counterTop)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 spatula))
  )

  (:goal (and
    (at-location spatula drawer1)
    (object-close robot1 drawer1)
  ))
)
```