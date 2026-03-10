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
  )
  (:goal
    (and
      (switch-off robot1 LightSwitch)
      (switch-off robot1 FloorLamp)
      (switch-off robot1 DeskLamp)
    )
  )
)