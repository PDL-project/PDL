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
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location mug diningtable)
    (at-location coffeemachine diningtable)
    (not (holding robot1 mug))
    ;; coffeemachine is non-openable, so no (object-close) needed
  )

  (:goal (and
    (at-location mug coffeemachine)
  ))

  (:metric minimize (total-cost))
)