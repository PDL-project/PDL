(define (problem open-seventh-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer7 - object
    kitchen - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location drawer7 floor)
    (object-close robot1 drawer7)
  )

  (:goal (and
    (object-open robot1 drawer7)
  ))

  (:metric minimize (total-cost))
)