(define (problem grabtomatoandgotofridge_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    tomato - object
    countertop - object
    fridge - object
    startinglocation - object
  )
  
  (:init
    (at robot1 startinglocation)
    (at-location tomato countertop)
    ;; Remove inaction since actions require not inaction as a precondition.
    (not (holding robot1 tomato))
  )
  
  (:goal
    (and
      (at robot1 fridge)
      (holding robot1 tomato)
    )
  )
)