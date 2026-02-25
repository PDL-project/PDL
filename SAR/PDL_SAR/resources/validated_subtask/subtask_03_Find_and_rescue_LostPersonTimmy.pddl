(define (problem find-and-rescue-lostperson-timmy)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    LostPersonTimmy - object
    DepositFacility - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 kitchen)
    (is-person LostPersonTimmy)
    (is-deposit DepositFacility)
    (not (person-found LostPersonTimmy))
    (not (carrying robot1 LostPersonTimmy))
  )

  (:goal (and
    (person-rescued LostPersonTimmy)
  ))

  (:metric minimize (total-cost))
)