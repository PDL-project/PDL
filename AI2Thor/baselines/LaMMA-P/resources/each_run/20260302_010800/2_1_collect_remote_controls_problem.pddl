(define (problem collect_remote_controls_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    RemoteControl1 - object
    RemoteControl2 - object
    Box - object
    SideTable - object
    CoffeeTable - object
    Floor - object
  )
  (:init
    (at robot2 Floor)
    (at-location RemoteControl1 SideTable)
    (at-location RemoteControl2 CoffeeTable)
    (at-location Box Floor)
  )
  (:goal
    (and
      (at-location RemoteControl1 Box)
      (at-location RemoteControl2 Box)
    )
  )
)