(define (problem rescue-lostpersonleo)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    base_camp - object
    LostPersonLeo - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 base_camp)
    (at robot2 base_camp)
    (is-person LostPersonLeo)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonLeo)
  ))

  (:metric minimize (total-cost))
)