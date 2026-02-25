(define (problem put-bread-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread - object
    fridge - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 countertop)
    (at-location bread countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 bread))
    (object-close robot1 fridge)
  )

  (:goal (and
    (at-location bread fridge)
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)