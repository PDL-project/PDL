(define (problem rescue-lostpersonethan)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    base_camp - object
    LostPersonEthan - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 base_camp)
    (at robot2 base_camp)
    (is-person LostPersonEthan)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonEthan)
  ))

  (:metric minimize (total-cost))
)