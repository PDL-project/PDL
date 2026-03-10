(define (problem move_remote_control_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    remoteControl - object
    coffeeTable - object
    couch - object
  )
  (:init
    (at robot2 couch)
    (at-location remoteControl couch)
  )
  (:goal
    (and
      (at-location remoteControl coffeeTable)
    )
  )
)