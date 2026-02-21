```pddl
(define (problem move-pan-to-stove)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    diningTable - object
    stoveBurner - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location pan diningTable)
    (not (holding robot1 pan))
  )

  (:goal (and
    (at-location pan stoveBurner)
    (not (holding robot1 pan))
  ))
)
```