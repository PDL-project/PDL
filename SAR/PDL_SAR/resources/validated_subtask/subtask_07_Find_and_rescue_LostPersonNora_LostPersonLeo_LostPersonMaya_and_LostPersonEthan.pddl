(define (problem find-and-rescue-lost-persons)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    LostPersonNora - object
    LostPersonLeo - object
    LostPersonMaya - object
    LostPersonEthan - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 kitchen)
    (is-person LostPersonNora)
    (is-person LostPersonLeo)
    (is-person LostPersonMaya)
    (is-person LostPersonEthan)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonNora)
    (person-rescued LostPersonLeo)
    (person-rescued LostPersonMaya)
    (person-rescued LostPersonEthan)
  ))

  (:metric minimize (total-cost))
)