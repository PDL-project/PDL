```pddl
(define (problem put-bread1-in-fridge1)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread1 - object
    fridge1 - object
    kitchen - object
    countertop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bread1 countertop)
    (at-location fridge1 floor)

    (is-fridge fridge1)
    (not (fridge-open fridge1))
    (not (holding robot1 bread1))
  )

  (:goal (and
    (at-location bread1 fridge1)
    (not (holding robot1 bread1))
  ))
)
```