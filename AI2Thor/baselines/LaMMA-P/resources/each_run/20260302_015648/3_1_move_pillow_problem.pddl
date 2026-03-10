(define (problem move_pillow_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    pillow - object
    armchair - object
    sofa - object
    floor - object
  )
  (:init
    (at robot3 sofa)
    (at-location pillow sofa)
    (at-location armchair floor)
  )
  (:goal
    (and
      (at-location pillow armchair)
    )
  )
)