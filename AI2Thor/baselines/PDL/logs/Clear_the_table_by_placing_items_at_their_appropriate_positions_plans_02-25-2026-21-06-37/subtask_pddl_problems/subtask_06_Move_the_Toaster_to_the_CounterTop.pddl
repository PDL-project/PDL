```pddl
(define (problem move-toaster-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    toaster - object
    diningTable - object
    counterTop - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location toaster diningTable)
    (at-location counterTop floor)
    (not (holding robot1 toaster))
  )

  (:goal (and
    (at-location toaster counterTop)
    (not (holding robot1 toaster))
  ))
)
```