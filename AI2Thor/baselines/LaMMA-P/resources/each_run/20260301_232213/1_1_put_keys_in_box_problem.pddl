(define (problem put_keys_in_box_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    keys - object
    box - object
    table - object
    shelf - object
  )
  (:init
    (at robot2 table)
    (at-location keys table)
    (at-location box shelf)
  )
  (:goal
    (and
      (at-location keys box)
    )
  )
)