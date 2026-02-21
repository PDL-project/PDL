```lisp
(define (problem open-drawer6)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer6 - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location drawer6 floor)
    (object-close robot1 drawer6)
  )

  (:goal (and
    (object-open robot1 drawer6)
  ))
)
```