(define (problem store-knife-in-bowl)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    bowl - object
    drawer1 - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (holding robot1 knife)
    (at-location bowl kitchen)
    (at-location drawer1 kitchen)
    (object-close robot1 drawer1)
  )

  (:goal (and
    (at-location knife bowl)
    (object-close robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)