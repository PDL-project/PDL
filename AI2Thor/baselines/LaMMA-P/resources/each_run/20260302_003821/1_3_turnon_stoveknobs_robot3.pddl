(define (problem turnon_stoveknobs_robot3)
  (:domain robot3)
  (:objects
    robot3 - robot
    stoveknob3 - object
    initiallocation - object
  )
  
  (:init 
    (at robot3 initiallocation)
    ; Ensure no inaction predicate is included in :init as per critical rules.
    ; Assume that the location of the knob is known and accessible.
   )

   (:goal 
     (switch-on robot3 stoveknob3) 
   ) 
)