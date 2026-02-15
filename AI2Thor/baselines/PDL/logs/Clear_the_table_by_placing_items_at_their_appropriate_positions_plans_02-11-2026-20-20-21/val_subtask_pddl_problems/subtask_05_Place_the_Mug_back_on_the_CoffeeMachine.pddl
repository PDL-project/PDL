(define (problem place-mug-on-coffeeMachine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    mug - object
    coffeeMachine - object
    diningTable - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningTable)
    (at-location mug diningTable)
    (at-location coffeeMachine diningTable)
    (not (holding robot1 mug))
  )

  (:goal (and
    (at-location mug coffeeMachine)
    (not (holding robot1 mug))
  ))

  (:metric minimize (total-cost))
)