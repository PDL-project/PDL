(define (problem store-the-fork)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fork - object
    drawer1 - object
    counterTop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location fork counterTop)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 fork))
  )

  (:goal (and
    (at-location fork drawer1)
    (object-close robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)