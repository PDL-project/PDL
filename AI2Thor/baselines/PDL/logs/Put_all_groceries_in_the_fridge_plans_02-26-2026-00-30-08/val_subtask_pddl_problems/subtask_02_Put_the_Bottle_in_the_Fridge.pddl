(define (problem put-bottle-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bottle - object
    fridge - object
    shelf - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bottle shelf)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 bottle))
    (object-close robot1 fridge)
  )

  (:goal (and
    (at-location bottle fridge)
    (not (holding robot1 bottle))
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)