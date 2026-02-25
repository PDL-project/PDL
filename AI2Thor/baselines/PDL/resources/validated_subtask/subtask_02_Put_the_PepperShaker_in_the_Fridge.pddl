(define (problem put-peppershaker-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    peppershaker - object
    fridge - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 countertop)
    (at-location peppershaker countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (object-close robot1 fridge)
    (not (holding robot1 peppershaker))
  )

  (:goal (and
    (at-location peppershaker fridge)
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)