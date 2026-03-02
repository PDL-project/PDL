```
(define (problem rescue-timmy)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    base_camp - object
    LostPersonTimmy - object
    DepositFacility - object
  )

  (:init
    (at robot1 base_camp)
    (at robot2 base_camp)
    (is-person LostPersonTimmy)
    (person-lost LostPersonTimmy)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (person-rescued LostPersonTimmy)
  ))
)
```