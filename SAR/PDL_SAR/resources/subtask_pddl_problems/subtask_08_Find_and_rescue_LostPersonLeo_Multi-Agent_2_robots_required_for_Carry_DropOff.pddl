```pddl
(define (problem find-and-rescue-lostpersonleo)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    LostPersonLeo - object
    DepositFacility - object
  )

  (:init
    (at robot1 kitchen)
    (not (person-found LostPersonLeo))
    (is-person LostPersonLeo)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonLeo)
  ))
)
```