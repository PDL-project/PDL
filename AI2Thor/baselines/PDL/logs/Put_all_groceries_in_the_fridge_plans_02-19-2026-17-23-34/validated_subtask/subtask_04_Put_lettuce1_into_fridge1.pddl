(define (problem put-lettuce-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    lettuce1 - object
    fridge1 - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 countertop)
    (at-location lettuce1 countertop)
    (at-location fridge1 floor)

    (is-fridge fridge1)
    (not (fridge-open fridge1))
    (not (holding robot1 lettuce1))
  )

  (:goal (and
    (at-location lettuce1 fridge1)
    (not (holding robot1 lettuce1))
    (object-close robot1 fridge1)
  ))

  (:metric minimize (total-cost))
)