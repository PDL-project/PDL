(define (problem put-saltshaker-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    saltshaker - object
    fridge - object
    countertop - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location saltshaker countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (object-close robot1 fridge)
    (not (holding robot1 saltshaker))
  )

  (:goal (and
    (at-location saltshaker fridge)
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)