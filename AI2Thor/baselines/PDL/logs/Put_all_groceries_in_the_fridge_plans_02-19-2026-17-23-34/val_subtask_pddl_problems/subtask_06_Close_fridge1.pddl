(define (problem close-fridge1)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fridge1 - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location fridge1 floor)

    (is-fridge fridge1)
    (fridge-open fridge1)
  )

  (:goal (and
    (object-close robot1 fridge1)
    (not (fridge-open fridge1))
  ))

  (:metric minimize (total-cost))
)