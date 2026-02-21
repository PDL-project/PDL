(define (problem move-bread-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread - object
    diningtable - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location bread diningtable)
    (at-location diningtable floor)
    (at-location countertop floor)
    (not (holding robot1 bread))
  )

  (:goal (and
    (at-location bread countertop)
    (not (holding robot1 bread))
  ))

  (:metric minimize (total-cost))
)