(define (problem open-fifth-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location drawer floor)
    (object-close robot1 drawer)
  )

  (:goal (and
    (object-open robot1 drawer)
  ))

  (:metric minimize (total-cost))
)