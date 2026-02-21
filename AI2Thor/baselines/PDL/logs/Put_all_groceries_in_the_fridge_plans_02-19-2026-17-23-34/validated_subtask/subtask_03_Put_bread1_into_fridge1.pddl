(define (problem put-bread1-in-fridge1)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread1 - object
    fridge1 - object
    kitchen - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bread1 countertop)
    (at-location fridge1 floor)

    (is-fridge fridge1)
    (not (fridge-open fridge1))
    (not (holding robot1 bread1))
    (object-close robot1 fridge1)
  )

  (:goal (and
    (at-location bread1 fridge1)
    (not (holding robot1 bread1))
    (object-close robot1 fridge1)
  ))

  (:metric minimize (total-cost))
)