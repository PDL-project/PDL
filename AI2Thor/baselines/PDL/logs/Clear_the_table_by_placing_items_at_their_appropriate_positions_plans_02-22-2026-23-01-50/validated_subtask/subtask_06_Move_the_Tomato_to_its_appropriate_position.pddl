(define (problem move-tomato-to-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    fridge - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location tomato diningtable)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 tomato))
  )

  (:goal (and
    (at-location tomato fridge)
    (not (holding robot1 tomato))
    (not (fridge-open fridge))
  ))

  (:metric minimize (total-cost))
)