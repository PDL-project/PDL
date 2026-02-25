(define (problem put_lettuce_in_fridge_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    lettuce - object
    fridge - object
    counterTop kitchenArea - object 
  )
  
  (:init 
    ; Initial state of the world 
    ; The location of objects and initial state of the robots
    
    ; Location of objects 
    ; Assuming that both are initially on a counter top in a kitchen area
    
  	(at-location lettuce counterTop)
  	(at-location fridge kitchenArea)

  	; Initial state of robots 
  	; Assuming that they are initially inactive
    
  	(not(in-action robot2))
    
  	; Other initial conditions can be added here if needed
    
  	...
    
   )
   
   (:goal 
     ; The goal is to have the lettuce inside the fridge
    
     ; Assuming that this means it is located at the same place as the fridge
    
     ; This can be adjusted based on more specific requirements or definitions of "inside"
    
     ; For now assuming it means same location as fridge
    
     ; This can be adjusted based on more specific requirements or definitions of "inside"
    
     ; For now assuming it means same location as fridge
    
     ...

     )
     
)