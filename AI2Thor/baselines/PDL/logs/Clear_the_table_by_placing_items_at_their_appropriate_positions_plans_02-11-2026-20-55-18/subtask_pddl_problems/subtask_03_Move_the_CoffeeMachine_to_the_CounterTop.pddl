```lisp
(define (problem move-coffeeMachine-to-counterTop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    coffeeMachine - object
    counterTop - object
    diningTable - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location coffeeMachine diningTable)
    (at-location counterTop floor)
    (not (holding robot1 coffeeMachine))
  )

  (:goal (and
    (at-location coffeeMachine counterTop)
    (not (holding robot1 coffeeMachine))
  ))
)
```