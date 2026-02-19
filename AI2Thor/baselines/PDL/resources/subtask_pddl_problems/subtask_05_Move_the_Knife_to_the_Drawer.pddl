```pddl
(define (problem move-knife-to-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    drawer - object
    sink - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 sink)
    (at-location knife sink)
    (at-location drawer floor)
    (not (holding robot1 knife))
    (object-close robot1 drawer)
  )

  (:goal (and
    (at-location knife drawer)
    (object-close robot1 drawer)
  ))
)
```