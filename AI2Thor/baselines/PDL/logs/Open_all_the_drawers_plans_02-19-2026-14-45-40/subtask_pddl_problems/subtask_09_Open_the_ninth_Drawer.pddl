```pddl
(define (problem open-ninth-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location drawer floor)
    (object-close robot1 drawer)
  )

  (:goal (and
    (object-open robot1 drawer)
  ))
)
```