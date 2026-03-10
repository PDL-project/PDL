(define (problem put_remote_in_box_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    RemoteControl - object
    Box - object
    SideTable - object
    Floor - object
  )
  (:init
    (at robot1 SideTable)
    (at-location RemoteControl SideTable)
    (at-location Box Floor)
  )
  (:goal
    (and
      (at-location RemoteControl Box)
    )
  )
)