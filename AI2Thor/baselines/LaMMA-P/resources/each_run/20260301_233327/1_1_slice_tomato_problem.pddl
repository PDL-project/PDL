(define (problem slice_tomato_problem)
  (:domain robot3)
  (:objects
    robot3 - robot
    tomato - object
    knife - object
    counterTop - object
  )
  (:init
    (at robot3 counterTop)
    (at-location tomato counterTop)
    (at-location knife counterTop)
    (not (inaction robot3)) ; Ensure the robot is not inaction to perform tasks.
  )
  (:goal
    (and
      (sliced tomato)
    )
  )
)