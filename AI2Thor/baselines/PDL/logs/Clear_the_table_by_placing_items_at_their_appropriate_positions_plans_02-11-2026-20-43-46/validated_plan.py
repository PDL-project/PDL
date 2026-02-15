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
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bread diningtable)
    (at-location countertop floor)
    (not (holding robot1 bread))
    (object-close robot1 countertop)
  )

  (:goal (and
    (at-location bread countertop)
    (not (holding robot1 bread))
    (object-close robot1 countertop)
  ))
)