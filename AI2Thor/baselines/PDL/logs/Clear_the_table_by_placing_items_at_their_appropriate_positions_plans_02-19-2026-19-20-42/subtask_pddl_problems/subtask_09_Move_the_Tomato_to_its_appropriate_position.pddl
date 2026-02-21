```pddl
(define (problem move-tomato-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    fridge - object
    diningtable - object
    kitchen - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location tomato diningtable)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 tomato))
  )

  (:goal (and
    (at-location tomato fridge)
    (not (holding robot1 tomato))
  ))
)
```