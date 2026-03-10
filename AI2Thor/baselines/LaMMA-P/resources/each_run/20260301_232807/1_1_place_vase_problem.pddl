(define (problem place_vase_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    vase - object
    sideTable - object
    diningTable - object
    floor - object
  )
  (:init
    (at robot1 floor)
    (at-location vase sideTable)
  )
  (:goal
    (and
      (at-location vase diningTable)
    )
  )
)