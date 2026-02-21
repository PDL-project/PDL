(define (problem place-spoon-in-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    drawer1 - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location spoon diningtable)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 spoon))
  )

  (:goal (and
    (at-location spoon drawer1)
    (object-close robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)