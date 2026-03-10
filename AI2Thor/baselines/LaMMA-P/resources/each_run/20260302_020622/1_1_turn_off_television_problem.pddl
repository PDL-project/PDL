(define (problem turn_off_television_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    television - object
    TVStand - object
  )
  (:init
    (at robot3 TVStand)
    (at-location television TVStand)
  )
  (:goal
    (and
      (switch-off robot3 television)
    )
  )
)