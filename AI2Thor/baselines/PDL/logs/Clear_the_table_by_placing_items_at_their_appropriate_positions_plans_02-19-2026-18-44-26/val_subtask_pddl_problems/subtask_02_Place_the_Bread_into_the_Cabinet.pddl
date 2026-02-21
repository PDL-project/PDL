(define (problem place-bread-into-cabinet)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread - object
    cabinet1 - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location bread diningtable)
    (at-location cabinet1 floor)
    (object-close robot1 cabinet1)
    (not (holding robot1 bread))
  )

  (:goal (and
    (at-location bread cabinet1)
    (object-close robot1 cabinet1)
  ))

  (:metric minimize (total-cost))
)