```pddl
(define (problem find-and-rescue-lostpersonethan)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    LostPersonEthan - object
    DepositFacility - object
  )

  (:init
    (at robot1 kitchen)
    (not (person-found LostPersonEthan))
    (is-person LostPersonEthan)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonEthan)
  ))
)
```