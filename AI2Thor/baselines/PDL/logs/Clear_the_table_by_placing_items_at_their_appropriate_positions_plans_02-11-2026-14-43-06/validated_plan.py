(define (problem move-mug-to-coffeemachine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    mug - object
    coffeemachine - object
    diningtable - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location mug diningtable)
    (not (holding robot1 mug))
  )

  (:goal (and
    (at-location mug coffeemachine)
  ))
)