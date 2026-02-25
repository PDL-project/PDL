(define (problem puttomatoinfridge_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    tomato - object
    fridge - object
    startinglocation - object
  )
  
  (:init
    (at robot2 startinglocation)
    (is-fridge fridge)
    (fridge-open fridge) ;; Assume initially closed, will be opened by action.
    (inaction robot2)
    (not(holding robot2 tomato))
  )
  
  (:goal
      (and 
        (at-location tomato fridge) 
        (object-close ?r ?f) ;; Fridge should be closed after placing the tomato.
      )
    
  )
)