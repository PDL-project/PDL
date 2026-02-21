(define (problem move-spoon-to-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    drawer1 - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)

    (at-location spoon diningtable)
    (at-location drawer1 floor)

    (not (holding robot1 spoon))
    (object-close robot1 drawer1)
  )

  (:goal (and
    (at-location spoon drawer1)
    (object-close robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)