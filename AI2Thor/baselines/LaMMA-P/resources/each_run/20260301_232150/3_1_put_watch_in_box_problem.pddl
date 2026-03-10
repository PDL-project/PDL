(define (problem put_watch_in_box_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    watch - object
    box - object
    SideTable - object
  )
  (:init
    (at robot3 SideTable)
    (at-location watch SideTable)
    (at-location box SideTable)
  )
  (:goal
    (and
      (at-location watch box)
    )
  )
)