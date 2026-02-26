```pddl
(define (problem store-butterknife)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    butterknife - object
    drawer1 - object
    countertop1 - object
    floor1 - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 floor1)
    (at-location butterknife countertop1)
    (at-location drawer1 floor1)
    (object-close robot1 drawer1)
  )

  (:goal (and
    (at-location butterknife drawer1)
    (object-close robot1 drawer1)
  ))
)
```