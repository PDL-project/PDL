(define (problem turnon_stoveknobs_robot1)
  (:domain robot1)
  (:objects
    robot1 - robot
    stoveknob1 - object
    stoveknob4 - object
    initiallocation - object
  )
  
  (:init
    (at robot1 initiallocation)
    (at-location stoveknob1 initiallocation)
    (at-location stoveknob4 initiallocation)
    ; Ensure no inaction predicate is included in :init as per critical rules.
  )
  
  (:goal
    (and
      (switch-on robot1 stoveknob1)
      (switch-on robot1 stoveknob4)
    )
  )
)