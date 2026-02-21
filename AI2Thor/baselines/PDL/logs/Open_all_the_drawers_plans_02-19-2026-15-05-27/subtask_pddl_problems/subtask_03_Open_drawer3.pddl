```lisp
(define (problem open-drawer3)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer3 - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location drawer3 floor)
    (object-close robot1 drawer3)
  )

  (:goal (and
    (object-open robot1 drawer3)
  ))
)
```