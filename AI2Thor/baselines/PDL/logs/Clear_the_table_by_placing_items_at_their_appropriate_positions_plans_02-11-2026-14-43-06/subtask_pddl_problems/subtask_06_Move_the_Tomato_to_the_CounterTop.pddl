```pddl
(define (problem move-tomato-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    diningTable - object
    counterTop - object
    kitchen - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location tomato diningTable)
    (at-location counterTop floor)
    (not (holding robot1 tomato))
  )

  (:goal (and
    (at-location tomato counterTop)
    (not (holding robot1 tomato))
  ))
)
```