(define (problem putTomatoInFridge_problem_robot1)
  (:domain robot1)
  (:objects
     robot1 - robot 
     tomato - object 
     fridge - object 
     initialLocationTomatoLocationFridgeLocation - object
   )
   (:init 
     ; Initial states based on the objects and preconditions above.
     (at-location tomato initialLocationTomatoLocationFridgeLocation)
     (at-location fridge initialLocationTomatoLocationFridgeLocation)
     (at robot1 initialLocationTomatoLocationFridgeLocation)
     (inaction robot1)
   )
   (:goal 
     ; Define goals based on the subtask examination above.
     (and
       (at-location tomato fridge)
       (not (holding robot1 tomato))
       (inaction robot1)
     )
   )
)