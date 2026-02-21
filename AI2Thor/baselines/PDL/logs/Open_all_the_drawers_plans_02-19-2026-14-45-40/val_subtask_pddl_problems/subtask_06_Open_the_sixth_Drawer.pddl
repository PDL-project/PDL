(define (problem open-sixth-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    drawer6 - object
    kitchen - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location drawer6 floor)
    (object-close robot1 drawer6)
  )

  (:goal (and
    (object-open robot1 drawer6)
    (object-close robot1 drawer6)
  ))

  (:metric minimize (total-cost))
)