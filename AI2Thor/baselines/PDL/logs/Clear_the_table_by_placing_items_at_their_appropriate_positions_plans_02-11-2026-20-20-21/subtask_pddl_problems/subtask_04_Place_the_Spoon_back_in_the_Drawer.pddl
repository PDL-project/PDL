```pddl
(define (problem place-spoon-in-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    drawer - object
    diningTable - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location spoon diningTable)
    (at-location drawer floor)
    (not (holding robot1 spoon))
    (object-close robot1 drawer)
  )

  (:goal (and
    (at-location spoon drawer)
    (object-close robot1 drawer)
  ))
)
```