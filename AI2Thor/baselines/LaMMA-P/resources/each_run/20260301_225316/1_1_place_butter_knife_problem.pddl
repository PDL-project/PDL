(define (problem place_butter_knife_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    ButterKnife - object
    CounterTop - object
    Drawer - object
    Floor - object
  )
  (:init
    (at robot1 Floor)
    (at-location ButterKnife Drawer)
  )
  (:goal
    (and
      (at-location ButterKnife CounterTop)
    )
  )
)