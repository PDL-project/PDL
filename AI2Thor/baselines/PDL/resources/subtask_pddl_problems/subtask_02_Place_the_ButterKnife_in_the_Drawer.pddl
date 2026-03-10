```pddl
(define (problem place-butterknife-in-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    butterKnife - object
    drawer1 - object
    diningTable - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)

    (at-location butterKnife diningTable)
    (at-location drawer1 floor)

    (not (holding robot1 butterKnife))
    (object-close robot1 drawer1)
  )

  (:goal (and
    (at-location butterKnife drawer1)
    (object-close robot1 drawer1)
  ))
)
```