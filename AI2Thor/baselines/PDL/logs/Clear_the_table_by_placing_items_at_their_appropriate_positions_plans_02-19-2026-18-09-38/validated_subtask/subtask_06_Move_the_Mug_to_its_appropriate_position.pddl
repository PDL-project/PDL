(define (problem move-mug-to-coffee-machine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    mug - object
    coffeeMachine - object
    diningTable - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location mug diningTable)
    (at-location coffeeMachine diningTable)
    (not (holding robot1 mug))
    (object-close robot1 coffeeMachine)
  )

  (:goal (and
    (at-location mug coffeeMachine)
    (not (holding robot1 mug))
    (object-close robot1 coffeeMachine)
  ))

  (:metric minimize (total-cost))
)