(define (problem turnon_stoveknobs_robot2)
  (:domain robot2)
  (:objects
    robot2 - robot
    stoveknob2 - object
    initiallocation - object
  )
  
  (:init
    (at robot2 initiallocation)
    ; Ensure no inaction predicate is included in :init as per critical rules.
    ; Assume that the location of the knob is known and accessible.
  )
  
  (:goal
    (switch-on robot2 stoveknob2)
  )
)