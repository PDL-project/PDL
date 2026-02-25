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
    ;; Assume initially closed, will be opened by action.
    ;; Remove (fridge-open fridge) if it's supposed to start closed.
    (inaction robot2)
    (not (holding robot2 tomato))
  )
  
  (:goal
      (and 
        (at-location tomato fridge) 
        (object-close robot2 fridge) ;; Fridge should be closed after placing the tomato.
      )
    
  )
)