(define (problem close-the-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer1 - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 drawer1)
    (at-location drawer1 floor)
    (object-open robot1 drawer1)
  )

  (:goal (and
    (object-close robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)