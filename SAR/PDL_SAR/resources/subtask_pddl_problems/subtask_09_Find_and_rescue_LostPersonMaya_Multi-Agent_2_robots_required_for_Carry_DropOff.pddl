```pddl
(define (problem rescue-lostpersonmaya)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    base_camp - object
    LostPersonMaya - object
    DepositFacility - object
  )

  (:init
    (at robot1 base_camp)
    (at robot2 base_camp)
    (is-person LostPersonMaya)
    (person-lost LostPersonMaya)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonMaya)
  ))
)
```