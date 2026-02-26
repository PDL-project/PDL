```pddl
(define (problem store-the-fork)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fork - object
    drawer1 - object
    countertop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location fork countertop)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 fork))
  )

  (:goal (and
    (at-location fork drawer1)
    (object-close robot1 drawer1)
  ))
)
```