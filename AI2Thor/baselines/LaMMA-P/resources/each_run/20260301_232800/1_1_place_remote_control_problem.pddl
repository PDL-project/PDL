(define (problem place_remote_control_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    RemoteControl - object
    Table - object
    Floor - object
    SideTable - object
  )
  (:init
    (at robot3 Floor) ; Initial location of the robot is Floor
    (at-location RemoteControl SideTable) ; Initial location of RemoteControl is SideTable
  )
  (:goal
    (and
      (at-location RemoteControl Table)
    )
  )
)