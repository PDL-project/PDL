```lisp
(define (problem store-the-knife)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    drawer1 - object
    countertop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location knife countertop)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 knife))
  )

  (:goal (and
    (at-location knife drawer1)
    (object-close robot1 drawer1)
  ))
)
```