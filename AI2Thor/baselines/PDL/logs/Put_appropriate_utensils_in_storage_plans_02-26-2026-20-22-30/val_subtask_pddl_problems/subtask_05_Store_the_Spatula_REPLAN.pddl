(define (problem store-spatula)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spatula - object
    drawer1 - object
    counterTop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 counterTop)
    (at-location spatula counterTop)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 spatula))
  )

  (:goal (and
    (at-location spatula drawer1)
    (object-close robot1 drawer1)
    (not (holding robot1 spatula))
  ))

  (:metric minimize (total-cost))
)