```pddl
(define (problem place-pan-in-cabinet)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    cabinet - object
    diningtable - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location pan diningtable)
    (at-location cabinet floor)
    (not (holding robot1 pan))
    (object-close robot1 cabinet)
  )

  (:goal (and
    (at-location pan cabinet)
    (object-close robot1 cabinet)
  ))
)
```