(define (problem rescue-timmy)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    base_camp - object
    LostPersonTimmy - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 base_camp)
    (at robot2 base_camp)
    (is-person LostPersonTimmy)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonTimmy)
  ))

  (:metric minimize (total-cost))
)