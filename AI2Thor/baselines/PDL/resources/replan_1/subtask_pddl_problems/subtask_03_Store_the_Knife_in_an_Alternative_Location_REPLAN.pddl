```lisp
(define (problem store-knife-in-alternative-location)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    drawer2 - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 floor)
    (holding robot1 knife)
    (at-location drawer2 floor)
    (object-close robot1 drawer2)
  )

  (:goal (and
    (at-location knife drawer2)
    (object-close robot1 drawer2)
  ))
)
```