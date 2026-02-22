```lisp
(define (problem move-mug-to-coffeemachine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    mug - object
    coffeemachine - object
    diningtable - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)

    (at-location mug diningtable)
    (at-location coffeemachine diningtable)

    (not (holding robot1 mug))
  )

  (:goal (and
    (at-location mug coffeemachine)
    (not (holding robot1 mug))
  ))
)
```