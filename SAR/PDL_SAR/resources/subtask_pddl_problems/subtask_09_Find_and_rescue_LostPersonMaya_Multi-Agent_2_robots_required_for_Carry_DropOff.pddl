```pddl
(define (problem find-and-rescue-lostpersonmaya)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    LostPersonMaya - object
    DepositFacility - object
  )

  (:init
    (not (person-found LostPersonMaya))
    (is-person LostPersonMaya)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonMaya)
  ))
)
```