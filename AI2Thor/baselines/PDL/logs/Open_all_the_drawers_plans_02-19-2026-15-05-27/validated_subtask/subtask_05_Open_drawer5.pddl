(define (problem open-drawer5)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer5 - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location drawer5 floor)
    (object-close robot1 drawer5)
  )

  (:goal (and
    (object-open robot1 drawer5)
  ))

  (:metric minimize (total-cost))
)