(define (problem turn_off_light_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    LightSwitch - object
    CounterTop - object
  )
  (:init
    (at robot2 CounterTop)
    (at-location LightSwitch CounterTop)
  )
  (:goal
    (and
      (switch-off robot2 LightSwitch)
    )
  )
)