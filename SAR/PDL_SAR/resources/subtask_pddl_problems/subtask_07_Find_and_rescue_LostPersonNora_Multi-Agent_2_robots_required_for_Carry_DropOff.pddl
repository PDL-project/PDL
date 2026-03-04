```lisp
(define (problem rescue-lostpersonnora)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    robot2 - robot
    base_camp - object
    LostPersonNora - object
    DepositFacility - object
  )

  (:init
    ; Both robots start at base camp — NEVER at LostPersonNora or DepositFacility.
    (at robot1 base_camp)
    (at robot2 base_camp)
    (is-person LostPersonNora)
    (person-lost LostPersonNora)
    (is-deposit DepositFacility)
  )

  (:goal (and
    (not (person-lost LostPersonNora))
  ))
)
```