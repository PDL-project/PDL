```lisp
(define (problem place-pan-into-cabinet)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    cabinet1 - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location pan diningtable)
    (at-location cabinet1 floor)
    (object-close robot1 cabinet1)
    (not (holding robot1 pan))
  )

  (:goal (and
    (at-location pan cabinet1)
    (object-close robot1 cabinet1)
  ))
)
```