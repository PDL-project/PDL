(define (problem put_plate_in_fridge_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    plate - object
    fridge - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at-location plate counterTop)
    (is-fridge fridge)
  )
  (:goal
    (and
      (at-location plate fridge)
    )
  )
)