(define (problem close_curtains_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    Curtains - object
    Window - object
  )
  (:init
    (at robot2 Window)
    (at-location Curtains Window)
  )
  (:goal
    (and
      (object-close robot2 Curtains)
    )
  )
)