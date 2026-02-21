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
    (not (inaction robot1))
    (at robot1 countertop)
    (at-location bread countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 bread))
  )

  (:goal (and
    (at-location bread fridge)
    (not (holding robot1 bread))
    (object-close robot1 fridge)
  ))
)