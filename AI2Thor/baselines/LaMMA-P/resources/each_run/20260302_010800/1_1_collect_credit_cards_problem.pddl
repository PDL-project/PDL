(define (problem collect_credit_cards_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    creditCard - object
    box - object
    sideTable - object
    floor - object
  )
  (:init
    (at robot1 sideTable)
    (at-location creditCard sideTable)
    (at-location box floor)
  )
  (:goal
    (and
      (at-location creditCard box)
    )
  )
)