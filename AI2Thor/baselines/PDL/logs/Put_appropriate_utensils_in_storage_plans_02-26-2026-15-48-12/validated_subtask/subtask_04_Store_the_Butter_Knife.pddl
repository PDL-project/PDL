(define (problem store-butter-knife)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    butterKnife - object
    drawer1 - object
    counterTop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location butterKnife counterTop)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 butterKnife))
  )

  (:goal (and
    (at-location butterKnife drawer1)
    (object-close robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)