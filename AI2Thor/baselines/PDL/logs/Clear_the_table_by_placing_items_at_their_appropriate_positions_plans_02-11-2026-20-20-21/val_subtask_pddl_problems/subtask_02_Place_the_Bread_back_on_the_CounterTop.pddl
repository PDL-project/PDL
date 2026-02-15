(define (problem place-bread-on-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread - object
    countertop - object
    diningtable - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location bread diningtable)
    (not (holding robot1 bread))
  )

  (:goal (and
    (at-location bread countertop)
    (not (holding robot1 bread))
  ))

  (:metric minimize (total-cost))
)