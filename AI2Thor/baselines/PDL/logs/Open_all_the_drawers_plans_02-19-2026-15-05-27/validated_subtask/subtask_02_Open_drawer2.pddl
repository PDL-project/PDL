(define (problem open-drawer2)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer2 - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location drawer2 floor)
    (object-close robot1 drawer2)
  )

  (:goal (and
    (object-open robot1 drawer2)
  ))

  (:metric minimize (total-cost))
)