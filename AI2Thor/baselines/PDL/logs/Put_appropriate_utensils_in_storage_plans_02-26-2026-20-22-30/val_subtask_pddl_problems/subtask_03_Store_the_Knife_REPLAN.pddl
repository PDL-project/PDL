(define (problem store-the-knife)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    drawer1 - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 drawer1)
    (holding robot1 knife)
    (object-open robot1 drawer1)
    (object-close robot1 drawer1)
  )

  (:goal (and
    (at-location knife drawer1)
    (object-close robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)