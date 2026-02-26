```pddl
(define (problem store-spatula)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spatula - object
    drawer1 - object
    countertop1 - object
    floor1 - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 floor1)
    (at-location spatula countertop1)
    (at-location drawer1 floor1)
    (object-close robot1 drawer1)
  )

  (:goal (and
    (at-location spatula drawer1)
    (object-close robot1 drawer1)
  ))
)
```