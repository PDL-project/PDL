(define (problem put_bread_in_fridge_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    bread - object
    fridge - object
    counterTop - object
    kitchen - object
  )
  (:init
    (at robot1 counterTop)
    (at-location bread counterTop)
    (at-location fridge kitchen)
    (inaction robot1)
  )
  (:goal
    (and
      (at-location bread fridge)
    )
  )
)