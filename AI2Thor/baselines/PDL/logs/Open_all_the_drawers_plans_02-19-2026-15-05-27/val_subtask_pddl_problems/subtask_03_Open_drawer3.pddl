(define (problem open-drawer3)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer3 - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 drawer3)
    (at-location drawer3 floor)
    (object-close robot1 drawer3)
  )

  (:goal (and
    (object-open robot1 drawer3)
  ))

  (:metric minimize (total-cost))
)