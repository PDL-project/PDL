```pddl
(define (problem move-lettuce-to-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    lettuce - object
    fridge - object
    diningtable - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location lettuce diningtable)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 lettuce))
  )

  (:goal (and
    (at-location lettuce fridge)
    (not (holding robot1 lettuce))
  ))
)
```