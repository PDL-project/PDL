```pddl
(define (problem find-and-rescue-lostpersonnora)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    LostPersonNora - object
    DepositFacility - object
  )

  (:init
    (not (person-found LostPersonNora))
    (not (carrying robot1 LostPersonNora))
    (not (carrying robot2 LostPersonNora))
  )

  (:goal (and
    (person-rescued LostPersonNora)
  ))
)
```