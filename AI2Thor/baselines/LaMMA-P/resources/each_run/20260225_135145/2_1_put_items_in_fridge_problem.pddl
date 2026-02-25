(define (problem put_items_in_fridge_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    lettuce - object
    tomato - object
    fridge - object
    counterTop - object
    kitchenArea - object
  )
  (:init
    (at robot2 counterTop)
    (at-location lettuce counterTop)
    (at-location tomato counterTop)
    (at-location fridge kitchenArea)
    (is-fridge fridge)
  )
  (:goal
    (and
      (in fridge lettuce)
      (in fridge tomato)
    )
  )
)