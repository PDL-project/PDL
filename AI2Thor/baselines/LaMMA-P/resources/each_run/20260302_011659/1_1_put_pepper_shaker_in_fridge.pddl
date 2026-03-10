(define (problem put_pepper_shaker_in_fridge)
  (:domain robot2)
  (:objects
    robot2 - robot
    PepperShaker - object
    Fridge - object
    CounterTop - object
    KitchenArea - object
  )
  (:init
    (at robot2 CounterTop)
    (at-location PepperShaker CounterTop)
    (at-location Fridge KitchenArea)
    (is-fridge Fridge)
  )
  (:goal
    (and
      (at-location PepperShaker Fridge)
    )
  )
)