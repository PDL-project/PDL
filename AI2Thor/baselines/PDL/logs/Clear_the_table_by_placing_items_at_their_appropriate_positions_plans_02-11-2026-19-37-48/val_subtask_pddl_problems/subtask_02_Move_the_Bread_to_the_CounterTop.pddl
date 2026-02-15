(define (problem move-bread-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread - object
    diningtable - object
    countertop - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location bread diningtable)
    (not (holding robot1 bread))
  )

  (:goal (and
    (at-location bread countertop)
    (not (holding robot1 bread))
  ))
)