```pddl
(define (problem open-drawer1)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer1 - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 floor)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
  )

  (:goal (and
    (object-open robot1 drawer1)
  ))
)
```