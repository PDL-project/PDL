(define (problem move-lettuce-to-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    lettuce - object
    fridge - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location lettuce diningtable)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 lettuce))
  )

  (:goal (and
    (at-location lettuce fridge)
    (not (holding robot1 lettuce))
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)