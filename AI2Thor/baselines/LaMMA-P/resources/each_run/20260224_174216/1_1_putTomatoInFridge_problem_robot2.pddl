(define (problem putTomatoInFridge_problem_robot2) 
   (:domain robot2) 
   (:objects 
      robot2- robot 
      tomato- object 
      fridge- object 
      initialLocationTomatoLocationFridgeLocation- object
   ) 

   (:init 
      (at-location tomato- initialLocationTomatoLocationFridgeLocation-)
      (at-location fridge- initialLocationTomatoLocationFridgeLocation-)
      (at robot2- initialLocationTomatoLocationFridgeLocation-)
      (not (inaction robot2-))
   ) 

   (:goal 
      (and
         (at-location tomato- fridge-)
         (cleaned robot2- fridge-) ; Assuming cleaning the fridge is part of the goal
         ; Add any other goals as needed
      )
   ) 
)