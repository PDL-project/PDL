(define (problem put-lettuce-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    lettuce - object
    fridge - object
    kitchen - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location lettuce countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (object-close robot1 fridge)
    (not (holding robot1 lettuce))
  )

  (:goal (and
    (at-location lettuce fridge)
    (object-close robot1 fridge)
    (not (holding robot1 lettuce))
  ))

  (:metric minimize (total-cost))
)