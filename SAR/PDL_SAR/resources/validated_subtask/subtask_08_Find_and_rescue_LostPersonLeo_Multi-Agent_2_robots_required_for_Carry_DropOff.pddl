(define (problem find-and-rescue-lostpersonleo)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    LostPersonLeo - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 kitchen)
    (is-person LostPersonLeo)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonLeo)
  ))

  (:metric minimize (total-cost))
)