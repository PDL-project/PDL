(define (problem move-mug-to-cabinet)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    mug - object
    diningTable - object
    cabinet1 - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location mug diningTable)
    (object-close robot1 cabinet1)
    (not (holding robot1 mug))
  )

  (:goal (and
    (at-location mug cabinet1)
    (object-close robot1 cabinet1)
  ))

  (:metric minimize (total-cost))
)