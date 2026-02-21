(define (problem put-tomato-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato1 - object
    fridge1 - object
    kitchen - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location tomato1 countertop)
    (at-location fridge1 floor)

    (is-fridge fridge1)
    (not (fridge-open fridge1))
    (object-close robot1 fridge1)
    (not (holding robot1 tomato1))
  )

  (:goal (and
    (at-location tomato1 fridge1)
    (object-close robot1 fridge1)
    (not (holding robot1 tomato1))
  ))

  (:metric minimize (total-cost))
)