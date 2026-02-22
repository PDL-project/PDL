```pddl
(define (problem move-spoon-to-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    drawer1 - object
    diningtable - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 floor)

    (at-location spoon diningtable)
    (at-location drawer1 floor)

    (object-close robot1 drawer1)
    (not (holding robot1 spoon))
  )

  (:goal (and
    (at-location spoon drawer1)
    (object-close robot1 drawer1)
  ))
)
```