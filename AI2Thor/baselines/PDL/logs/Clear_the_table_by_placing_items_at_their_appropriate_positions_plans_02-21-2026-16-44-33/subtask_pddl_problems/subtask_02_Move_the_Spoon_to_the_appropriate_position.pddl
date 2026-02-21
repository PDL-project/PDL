```lisp
(define (problem move-spoon-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    diningtable - object
    countertop - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location spoon diningtable)
    (not (holding robot1 spoon))
  )

  (:goal (and
    (at-location spoon countertop)
  ))
)
```