(define (problem find-and-rescue-lostpersonethan)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    LostPersonEthan - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 kitchen)
    (is-person LostPersonEthan)
    (is-deposit DepositFacility)
    (not (person-found LostPersonEthan))
  )

  (:goal (and
    (person-rescued LostPersonEthan)
  ))

  (:metric minimize (total-cost))
)