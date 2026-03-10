(define (problem turn_off_lights_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    LightSwitch - object
    FloorLamp - object
    DeskLamp - object
    Floor - object
  )
  (:init
    (at robot1 Floor)
    (at-location LightSwitch Floor)
    (at-location FloorLamp Floor)
    (at-location DeskLamp Floor)
    (not (inaction robot1)) ; Assuming this means the robot starts active.
  )
  (:goal
    (and
      (switch-off robot1 LightSwitch) ; Assuming this is correct based on domain definition.
      (switch-off robot1 FloorLamp)
      (switch-off robot1 DeskLamp)
    )
  )
)