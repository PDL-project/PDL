(define (problem turn_off_faucet_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    Faucet - object
    Sink - object
    Floor - object ; Added Floor as an object
  )
  (:init
    (at robot1 Floor)
    (at-location Faucet Sink)
  )
  (:goal
    (and
      (switch-off robot1 Faucet)
    )
  )
)