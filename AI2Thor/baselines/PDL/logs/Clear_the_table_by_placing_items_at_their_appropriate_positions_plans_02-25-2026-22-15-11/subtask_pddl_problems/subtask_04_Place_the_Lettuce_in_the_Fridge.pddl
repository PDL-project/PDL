```lisp
(define (problem place-lettuce-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    lettuce - object
    fridge - object
    diningTable - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location lettuce diningTable)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 lettuce))
  )

  (:goal (and
    (at-location lettuce fridge)
    (not (holding robot1 lettuce))
    (object-close robot1 fridge)
  ))
)
```