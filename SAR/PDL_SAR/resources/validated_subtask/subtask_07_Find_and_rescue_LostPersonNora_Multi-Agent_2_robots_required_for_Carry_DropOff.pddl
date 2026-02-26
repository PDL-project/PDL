(define (problem find-and-rescue-lostpersonnora)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    LostPersonNora - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (is-person LostPersonNora)
    (is-deposit DepositFacility)
    (not (person-found LostPersonNora))
    (not (carrying robot1 LostPersonNora))
    (not (carrying robot2 LostPersonNora))
  )

  (:goal (and
    (person-rescued LostPersonNora)
  ))

  (:metric minimize (total-cost))
)