```pddl
(define (problem move-mug-to-coffee-machine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    mug - object
    diningTable - object
    coffeeMachine - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location mug diningTable)
    (not (holding robot1 mug))
  )

  (:goal (and
    (at-location mug coffeeMachine)
    (not (holding robot1 mug))
  ))
)
```