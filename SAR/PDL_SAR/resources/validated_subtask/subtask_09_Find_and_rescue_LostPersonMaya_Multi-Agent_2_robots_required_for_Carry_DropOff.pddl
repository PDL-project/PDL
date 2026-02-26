(define (problem find-and-rescue-lostpersonmaya)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    LostPersonMaya - object
    DepositFacility - object
  )

  (:init
    (= (total-cost) 0)
    (is-person LostPersonMaya)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonMaya)
  ))

  (:metric minimize (total-cost))
)