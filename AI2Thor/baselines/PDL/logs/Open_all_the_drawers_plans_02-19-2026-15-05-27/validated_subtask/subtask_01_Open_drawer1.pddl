(define (problem open-drawer1)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer1 - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
  )

  (:goal (and
    (object-open robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)