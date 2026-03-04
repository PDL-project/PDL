(define (problem rescue-lostpersonnora)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    base_camp - object
    LostPersonNora - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 base_camp)
    (at robot2 base_camp)
    (is-person LostPersonNora)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonNora)
  ))

  (:metric minimize (total-cost))
)