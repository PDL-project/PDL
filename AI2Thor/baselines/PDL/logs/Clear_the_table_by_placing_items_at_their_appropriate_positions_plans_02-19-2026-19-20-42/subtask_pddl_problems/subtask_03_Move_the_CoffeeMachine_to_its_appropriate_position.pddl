```pddl
(define (problem move-coffee-machine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    coffeeMachine - object
    diningTable - object
    counterTop - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location coffeeMachine diningTable)
    (not (holding robot1 coffeeMachine))
  )

  (:goal (and
    (at-location coffeeMachine counterTop)
  ))
)
```