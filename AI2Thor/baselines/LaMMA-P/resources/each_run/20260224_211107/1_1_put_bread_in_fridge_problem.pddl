(define (problem put_bread_in_fridge_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    bread - object
    fridge - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at-location bread counterTop)
    (at-location fridge counterTop)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location bread fridge)
    )
  )
)