(define (problem open-drawer4)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer4 - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location drawer4 floor)
    (object-close robot1 drawer4)
  )

  (:goal (and
    (object-open robot1 drawer4)
  ))

  (:metric minimize (total-cost))
)