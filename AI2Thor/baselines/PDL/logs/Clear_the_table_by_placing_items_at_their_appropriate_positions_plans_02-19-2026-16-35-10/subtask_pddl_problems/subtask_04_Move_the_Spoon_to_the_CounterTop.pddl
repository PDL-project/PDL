```lisp
(define (problem move-spoon-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    diningtable - object
    countertop - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location spoon diningtable)
    (at-location countertop floor)
    (not (holding robot1 spoon))
  )

  (:goal (and
    (at-location spoon countertop)
    (not (holding robot1 spoon))
  ))
)
```