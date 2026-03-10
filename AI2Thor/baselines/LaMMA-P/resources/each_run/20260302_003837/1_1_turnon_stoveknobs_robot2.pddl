(define (problem turnon_stoveknobs_robot2)
  (:domain robot2)
  (:objects
    robot2 - robot
    stoveknob2 - object
    initiallocation - object
    stoveknob_location - object ; Added location for stoveknob
  )
  
  (:init
    (at robot2 initiallocation)
    (at-location stoveknob2 stoveknob_location) ; Specify location of stoveknob
    (not (inaction robot2)) ; Ensure robot is not in an "inaction" state
  )
  
  (:goal
    (switch-on robot2 stoveknob2)
  )
)