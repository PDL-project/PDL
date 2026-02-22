(define (problem move-mug-to-coffeemachine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    mug - object
    coffeemachine - object
    diningtable - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location mug diningtable)
    (at-location coffeemachine diningtable)
    (object-close robot1 coffeemachine) ;; CoffeeMachine is openable, starts closed
    (not (holding robot1 mug))
  )

  (:goal (and
    (at-location mug coffeemachine)
    (not (holding robot1 mug))
    (object-close robot1 coffeemachine) ;; Ensure CoffeeMachine is closed after placing the mug
  ))

  (:metric minimize (total-cost))
)