(define (problem put_saltshaker_in_fridge_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    SaltShaker - object
    fridge - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at-location SaltShaker counterTop)
    (at-location fridge counterTop)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location SaltShaker fridge)
    )
  )
)