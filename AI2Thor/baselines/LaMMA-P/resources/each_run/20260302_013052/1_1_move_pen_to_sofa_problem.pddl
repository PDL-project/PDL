(define (problem move_pen_to_sofa_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    pen - object
    sofa - object
    table - object
  )
  (:init
    (at robot2 table)
    (at-location pen table)
  )
  (:goal
    (and
      (at-location pen sofa)
    )
  )
)