(define (problem place_remote_control_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    RemoteControl - object
    Table - object
  )
  (:init
    (at robot3 Floor) ; Assuming initial location of the robot is Floor
    (at-location RemoteControl SideTable) ; Assuming initial location of RemoteControl is SideTable
  )
  (:goal
    (and
      (at-location RemoteControl Table)
    )
  )
)