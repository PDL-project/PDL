(define (problem move-fork-to-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fork - object
    drawer1 - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location fork diningtable)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 fork))
  )

  (:goal (and
    (at-location fork drawer1)
    (object-close robot1 drawer1)
  ))
)