(define (problem move_vase_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    vase - object
    floor - object
    countertop - object
  )
  (:init
    (at robot1 floor)
    (at-location vase floor)
  )
  (:goal
    (and
      (at-location vase countertop)
    )
  )
)