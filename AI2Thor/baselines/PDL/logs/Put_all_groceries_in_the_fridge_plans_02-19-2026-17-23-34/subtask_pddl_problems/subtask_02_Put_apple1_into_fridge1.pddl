```pddl
(define (problem put-apple1-in-fridge1)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    apple1 - object
    fridge1 - object
    kitchen - object
    countertop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location apple1 countertop)
    (at-location fridge1 floor)

    (is-fridge fridge1)
    (not (fridge-open fridge1))
    (not (holding robot1 apple1))
  )

  (:goal (and
    (at-location apple1 fridge1)
    (not (holding robot1 apple1))
  ))
)
```