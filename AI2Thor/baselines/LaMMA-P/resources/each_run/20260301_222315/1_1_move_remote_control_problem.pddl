(define (problem move_remote_control_problem)
  (:domain robot3)
  (:objects
    robo3 - robot
    remoteControl - object
    sofa - object
    sideTable - object
  )
  (:init
    (at robo3 sideTable)
    (at-location remoteControl sideTable)
  )
  (:goal
    (and
      (at-location remoteControl sofa)
    )
  )
)